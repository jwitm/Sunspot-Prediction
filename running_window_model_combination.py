"""Select and combine running-window models using mean Brier score.

The script reads time-resolved predictions from an HDF5 archive, reconstructs
each model's cross-validation split, finds the best individual model, and tests
non-empty model subsets for an ensemble with a lower mean Brier score. Paths and
compute settings are supplied through command-line arguments so the analysis can
run outside the original computing environment.
"""

import argparse
import multiprocessing as mp
import os
from itertools import combinations
from math import comb
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm


DEFAULT_OBSERVABLES = ('Dopplergram', 'Vec-Mag', 'Continuum', 'Magnetogram')
_WORKER_CONTEXT = {}


class Subsets:
    """Iterate over non-empty subsets up to a configurable maximum size."""

    def __init__(self, elements, max_size):
        """Initialize a lazily generated collection of subsets.

        Args:
            elements: Values from which subsets are constructed.
            max_size: Largest subset size to generate.
        """
        self.elements = elements
        self.n = len(elements)
        self.max_size = max_size
        self._len = sum(
            comb(self.n, size)
            for size in range(1, min(self.n, self.max_size) + 1)
        )

    def __iter__(self):
        """Yield every non-empty subset in increasing size order."""
        for size in range(1, min(self.n, self.max_size) + 1):
            yield from combinations(self.elements, size)

    def __len__(self):
        """Return the total number of generated subsets."""
        return self._len


def total_brier_score(predictions, reference):
    """Calculate the mean Brier score for one running-window sequence.

    For positive sequences, the final 15 predictions are excluded because those
    windows include frames in which a sunspot is already present.

    Args:
        predictions: One-dimensional tensor of predicted sunspot probabilities.
        reference: Binary sequence label.

    Returns:
        A scalar tensor, or NaN when a positive sequence has at most 15 windows.
    """
    if reference == 1:
        if len(predictions) > 15:
            return torch.mean((predictions[:-15] - reference) ** 2)
        return torch.tensor(float('nan'))
    return torch.mean((predictions - reference) ** 2)


def get_all_models(filename, expected_sequences, expected_type):
    """Return models containing the expected number of prediction sequences.

    The prediction HDF5 hierarchy is expected to be
    ``observable/correction/model/sequence_index``.
    """
    all_models = []
    with h5py.File(filename, 'r') as file:
        if expected_type not in file:
            return all_models
        type_group = file[expected_type]
        for correction in type_group.keys():
            correction_group = type_group[correction]
            for model_name in correction_group.keys():
                if len(correction_group[model_name].keys()) == expected_sequences:
                    all_models.append((correction, model_name))
    return all_models


def _find_model_path(models_dir, model_name):
    """Find a model checkpoint recursively below ``models_dir``."""
    matches = sorted(Path(models_dir).rglob(f'{model_name}.pth'))
    if not matches:
        raise FileNotFoundError(
            f'Model {model_name}.pth was not found below {models_dir}'
        )
    return matches[0]


def filter_validation(model, train, labels_dir, long_sequences_path, models_dir):
    """Map a model's cross-validation split to running-window sequence indices.

    Args:
        model: ``(correction, model_name)`` tuple.
        train: Select training HARPs when true and validation HARPs otherwise.
        labels_dir: Directory containing fixed-length sequence descriptor files.
        long_sequences_path: Descriptor file for the running-window sequences.
        models_dir: Root directory containing model checkpoints.

    Returns:
        Indices of long sequences whose HARPs belong to the selected split.
    """
    correction, model_name = model
    fold_to_select = int(model_name.split('_')[0])
    if fold_to_select not in range(5):
        raise ValueError(f'Cannot infer a fold from model name {model_name!r}')

    sequence_path = Path(labels_dir) / (
        f'Sequences_ws_23p8_corr_{correction}_filter_[15, 15].pt'
    )
    sequences = torch.load(
        sequence_path,
        weights_only=False,
        map_location=torch.device('cpu'),
    )
    long_sequences = torch.load(
        long_sequences_path,
        weights_only=False,
        map_location=torch.device('cpu'),
    )

    model_path = _find_model_path(models_dir, model_name)
    checkpoint = torch.load(
        model_path,
        map_location=torch.device('cpu'),
        weights_only=False,
    )
    random_state = checkpoint['parameters']['random_state']

    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
    valid_harps = set()
    for fold, (train_indices, validation_indices) in enumerate(
        kfold.split(np.arange(len(sequences)))
    ):
        if fold == fold_to_select:
            selected = train_indices if train else validation_indices
            valid_harps = {sequences[index]['Harp'] for index in selected}
            break

    return [
        index
        for index, item in enumerate(long_sequences)
        if item['Harp'] in valid_harps
    ]


def get_baseline_performance(
    filename,
    expected_sequences,
    reference,
    model_valid_indices,
    expected_type='Continuum',
):
    """Find the eligible individual model with the lowest mean Brier score.

    Returns:
        The correction, model name, best score, and complete score table.
    """
    performance = {}
    with h5py.File(filename, 'r') as file:
        if expected_type not in file:
            raise KeyError(f'Prediction group {expected_type!r} is not in {filename}')
        type_group = file[expected_type]
        for correction in type_group.keys():
            correction_group = type_group[correction]
            for model_name in correction_group.keys():
                model_group = correction_group[model_name]
                if len(model_group.keys()) != expected_sequences:
                    continue

                model_key = (correction, model_name)
                scores = []
                for index in model_valid_indices[model_key]:
                    predictions = torch.from_numpy(
                        np.asarray(model_group[str(index)]['predictions'])[:, -1]
                    )
                    score = total_brier_score(
                        predictions,
                        reference[int(index)]['Label'],
                    )
                    scores.append(float(score))
                if scores and not np.all(np.isnan(scores)):
                    performance[model_key] = float(np.nanmean(scores))

    if not performance:
        raise ValueError('No eligible individual models were found')

    performance_dict = pd.DataFrame(
        [
            {'Correction': key[0], 'Model': key[1], 'Brier Score': score}
            for key, score in performance.items()
        ]
    ).set_index(['Correction', 'Model'])
    baseline_key = min(performance, key=performance.get)
    return (*baseline_key, performance[baseline_key], performance_dict)


def load_data_into_memory(filename, expected_sequences, expected_type):
    """Load eligible model predictions and uncertainties into CPU memory."""
    predictions_map = {}
    with h5py.File(filename, 'r') as file:
        if expected_type not in file:
            raise KeyError(f'Prediction group {expected_type!r} is not in {filename}')
        type_group = file[expected_type]
        for correction in tqdm(type_group.keys(), desc='Loading predictions'):
            correction_group = type_group[correction]
            for model_name in correction_group.keys():
                model_group = correction_group[model_name]
                if len(model_group.keys()) != expected_sequences:
                    continue
                prediction_list = []
                uncertainty_list = []
                for index in range(expected_sequences):
                    index_group = model_group[str(index)]
                    prediction_list.append(
                        torch.from_numpy(np.asarray(index_group['predictions'])[:, -1])
                    )
                    uncertainty_list.append(
                        torch.from_numpy(np.asarray(index_group['uncertainty'])[:, -1])
                    )
                predictions_map[(correction, model_name)] = {
                    'pred_list': prediction_list,
                    'unc_list': uncertainty_list,
                }
    return predictions_map


def get_model_validation_indices(
    all_models,
    labels_dir,
    long_sequences_path,
    models_dir,
    train=True,
):
    """Precalculate eligible running-window indices for every model."""
    model_valid_indices = {}
    for model in tqdm(all_models, desc='Calculating valid indices per model'):
        model_valid_indices[model] = set(
            filter_validation(
                model,
                train,
                labels_dir,
                long_sequences_path,
                models_dir,
            )
        )
    return model_valid_indices


def _initialize_worker(predictions, model_valid_indices, sequences, expected_sequences):
    """Install shared ensemble-scoring context in a worker process."""
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        'predictions': predictions,
        'model_valid_indices': model_valid_indices,
        'sequences': sequences,
        'expected_sequences': expected_sequences,
    }


def _process_subset(indexed_subset):
    """Score one model subset using the initialized worker context."""
    subset_index, subset = indexed_subset
    predictions = _WORKER_CONTEXT['predictions']
    model_valid_indices = _WORKER_CONTEXT['model_valid_indices']
    sequences = _WORKER_CONTEXT['sequences']
    expected_sequences = _WORKER_CONTEXT['expected_sequences']

    scores = []
    for sequence_index in range(expected_sequences):
        eligible_predictions = [
            predictions[model]['pred_list'][sequence_index]
            for model in subset
            if sequence_index in model_valid_indices[model]
        ]
        if not eligible_predictions:
            continue
        averaged = torch.mean(torch.vstack(eligible_predictions), dim=0)
        scores.append(
            float(total_brier_score(averaged, sequences[sequence_index]['Label']))
        )
    score = float(np.nanmean(scores)) if scores else float('inf')
    return subset_index, subset, score


def combine_predictions(
    subset,
    predictions,
    model_valid_indices,
    expected_sequences,
):
    """Combine predictions and uncertainties for the selected model subset."""
    combined = []
    for sequence_index in range(expected_sequences):
        prediction_tensors = []
        uncertainty_tensors = []
        for model in subset:
            if sequence_index in model_valid_indices[model]:
                prediction_tensors.append(
                    predictions[model]['pred_list'][sequence_index]
                )
                uncertainty_tensors.append(
                    predictions[model]['unc_list'][sequence_index]
                )
        if not prediction_tensors:
            combined.append(None)
            continue

        prediction_stack = torch.vstack(prediction_tensors)
        uncertainty_stack = torch.vstack(uncertainty_tensors)
        averaged_predictions = torch.mean(prediction_stack, dim=0)
        mean_of_variances = torch.mean(uncertainty_stack ** 2, dim=0)
        variance_of_means = torch.var(prediction_stack, dim=0, unbiased=False)
        combined_uncertainty = torch.sqrt(mean_of_variances + variance_of_means)
        combined.append(
            {
                'pred_list': averaged_predictions,
                'unc_list': combined_uncertainty,
            }
        )
    return combined


def _parse_args():
    """Parse command-line options for running-window ensemble selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions_file', type=Path, help='Running-window prediction HDF5 file')
    parser.add_argument('long_sequences', type=Path, help='Running-window sequence descriptor .pt file')
    parser.add_argument('labels_dir', type=Path, help='Directory with fixed-length sequence descriptor files')
    parser.add_argument('models_dir', type=Path, help='Directory containing transformer .pth checkpoints')
    parser.add_argument('output_dir', type=Path, help='Directory for combined predictions and summaries')
    parser.add_argument(
        '--expected-sequences',
        type=int,
        default=os.environ.get(
            'EXPECTED_SEQUENCES',
            os.environ.get('expected_sequences'),
        ),
    )
    parser.add_argument('--observable', choices=DEFAULT_OBSERVABLES)
    parser.add_argument('--prediction-group', help='Exact observable group name in the HDF5 file')
    parser.add_argument('--task-id', type=int, default=os.environ.get('SLURM_ARRAY_TASK_ID'))
    parser.add_argument('--processes', type=int, default=os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    parser.add_argument('--max-subset-size', type=int, help='Limit ensemble size; default tests every size')
    parser.add_argument('--validation', action='store_true', help='Score validation rather than training HARPs')
    args = parser.parse_args()

    if args.expected_sequences is None:
        parser.error('--expected-sequences or the expected_sequences environment variable is required')
    args.expected_sequences = int(args.expected_sequences)
    args.processes = int(args.processes)
    if args.expected_sequences < 1:
        parser.error('--expected-sequences must be positive')
    if args.processes < 1:
        parser.error('--processes must be positive')
    if args.max_subset_size is not None and args.max_subset_size < 1:
        parser.error('--max-subset-size must be positive')

    if args.observable is None:
        if args.task_id is None:
            parser.error('--observable or --task-id/SLURM_ARRAY_TASK_ID is required')
        try:
            args.observable = DEFAULT_OBSERVABLES[int(args.task_id)]
        except (IndexError, ValueError):
            parser.error(f'task ID must be between 0 and {len(DEFAULT_OBSERVABLES) - 1}')
    if args.prediction_group is None:
        args.prediction_group = args.observable
    return args


def main():
    """Run exhaustive model-subset selection and save the best ensemble."""
    args = _parse_args()
    sequences = torch.load(
        args.long_sequences,
        map_location=torch.device('cpu'),
        weights_only=False,
    )
    if len(sequences) != args.expected_sequences:
        raise ValueError(
            f'Expected {args.expected_sequences} sequences but loaded {len(sequences)}'
        )

    all_models = get_all_models(
        args.predictions_file,
        args.expected_sequences,
        args.prediction_group,
    )
    if not all_models:
        raise ValueError(
            f'No eligible models found in prediction group {args.prediction_group!r}'
        )

    model_valid_indices = get_model_validation_indices(
        all_models,
        args.labels_dir,
        args.long_sequences,
        args.models_dir,
        train=not args.validation,
    )
    baseline_correction, baseline_model, baseline_score, performance = (
        get_baseline_performance(
            args.predictions_file,
            args.expected_sequences,
            sequences,
            model_valid_indices,
            args.prediction_group,
        )
    )
    baseline_key = (baseline_correction, baseline_model)
    print(f'Baseline: {baseline_key} with Brier score {baseline_score:.6f}')

    predictions = load_data_into_memory(
        args.predictions_file,
        args.expected_sequences,
        args.prediction_group,
    )
    max_subset_size = args.max_subset_size or len(all_models)
    subsets = Subsets(all_models, max_subset_size)
    best_score = baseline_score
    best_subset = (baseline_key,)

    initializer_args = (
        predictions,
        model_valid_indices,
        sequences,
        args.expected_sequences,
    )
    if args.processes == 1:
        _initialize_worker(*initializer_args)
        results = map(_process_subset, enumerate(subsets))
        for _, subset, score in tqdm(results, total=len(subsets), desc='Scoring subsets'):
            if score < best_score:
                best_score, best_subset = score, subset
    else:
        with mp.Pool(
            processes=args.processes,
            initializer=_initialize_worker,
            initargs=initializer_args,
        ) as pool:
            results = pool.imap(_process_subset, enumerate(subsets))
            for _, subset, score in tqdm(results, total=len(subsets), desc='Scoring subsets'):
                if score < best_score:
                    best_score, best_subset = score, subset

    combined_predictions = combine_predictions(
        best_subset,
        predictions,
        model_valid_indices,
        args.expected_sequences,
    )
    improvement = (
        (1 - best_score / baseline_score) * 100
        if baseline_score != 0
        else 0.0
    )
    print(f'Best subset: {best_subset}')
    print(f'Combined Brier score: {best_score:.6f}')
    print(f'Relative improvement: {improvement:.2f}%')

    observable_dir = args.output_dir / args.observable
    observable_dir.mkdir(parents=True, exist_ok=True)
    torch.save(combined_predictions, observable_dir / 'predictions.pt')
    performance.to_csv(observable_dir / 'individual_model_scores.csv')

    summary = pd.DataFrame(
        [
            {
                'Observable': args.observable,
                'Prediction Group': args.prediction_group,
                'Models': best_subset,
                'Baseline Correction': baseline_correction,
                'Baseline Model': baseline_model,
                'Baseline Score': baseline_score,
                'Combined Score': best_score,
                'Improvement (%)': improvement,
                'Split': 'validation' if args.validation else 'training',
            }
        ]
    )
    torch.save(summary, observable_dir / 'summary.pt')
    csv_summary = summary.copy()
    csv_summary['Models'] = csv_summary['Models'].map(repr)
    csv_summary.to_csv(observable_dir / 'summary.csv', index=False)


if __name__ == '__main__':
    main()
