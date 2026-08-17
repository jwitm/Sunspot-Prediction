"""Evaluate a selected running-window model ensemble on validation HARPs.

This script loads the model subset chosen by
``running_window_model_combination.py``, combines its previously generated
running-window predictions on the held-out cross-validation folds, and compares
the resulting Brier score with that of the training-selected baseline model. It
does not execute the neural networks; the per-model predictions must already be
available in the input HDF5 archive.
"""

import argparse
import ast
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from running_window_model_combination import (
    DEFAULT_OBSERVABLES,
    combine_predictions,
    get_all_models,
    get_baseline_performance,
    get_model_validation_indices,
    total_brier_score,
)


OBSERVABLE_CONFIG = {
    'Continuum': ('continuum', 'continuum'),
    'Dopplergram': ('Dopplergram', 'dopplergram'),
    'Magnetogram': ('magnetogram', 'magnetogram'),
    'Vec-Mag': (['Br', 'Bp', 'Bt'], 'vec-mag'),
}


def convert_name_to_data_type(name: str):
    """Convert a display observable name to dataset and HDF5 group formats.

    Args:
        name: One of ``Continuum``, ``Dopplergram``, ``Magnetogram``, or
            ``Vec-Mag``.

    Returns:
        A tuple containing the dataset observable value and default prediction
        HDF5 group name.

    Raises:
        ValueError: If ``name`` is not a supported observable.
    """
    try:
        return OBSERVABLE_CONFIG[name]
    except KeyError as error:
        valid = ', '.join(OBSERVABLE_CONFIG)
        raise ValueError(f'Invalid observable {name!r}; choose from {valid}') from error


def load_selected_predictions(
    filename,
    models,
    expected_type,
    expected_sequences,
):
    """Load predictions and uncertainties for a selected set of models.

    Args:
        filename: Running-window prediction HDF5 archive.
        models: Iterable of ``(correction, model_name)`` pairs.
        expected_type: Exact observable group name in the HDF5 archive.
        expected_sequences: Number of indexed sequences expected per model.

    Returns:
        Mapping from model pairs to prediction and uncertainty tensor lists.
    """
    predictions_map = {}
    with h5py.File(filename, 'r') as file:
        if expected_type not in file:
            raise KeyError(f'Prediction group {expected_type!r} is not in {filename}')
        type_group = file[expected_type]
        for correction, model_name in models:
            if correction not in type_group:
                raise KeyError(
                    f'Correction {correction!r} is missing from group {expected_type!r}'
                )
            correction_group = type_group[correction]
            if model_name not in correction_group:
                raise KeyError(
                    f'Model {model_name!r} is missing below correction {correction!r}'
                )
            model_group = correction_group[model_name]
            if len(model_group.keys()) != expected_sequences:
                raise ValueError(
                    f'Model {(correction, model_name)} contains '
                    f'{len(model_group.keys())} sequences; expected {expected_sequences}'
                )

            prediction_list = []
            uncertainty_list = []
            for sequence_index in range(expected_sequences):
                index_group = model_group[str(sequence_index)]
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


def _normalize_models(value):
    """Normalize serialized model combinations to tuples of model-key pairs."""
    if isinstance(value, str):
        value = ast.literal_eval(value)
    models = tuple(tuple(str(part) for part in model) for model in value)
    if not models or any(len(model) != 2 for model in models):
        raise ValueError(f'Invalid model combination: {value!r}')
    return models


def load_combination_summary(summary_path, observable):
    """Read selected models and an optional baseline from a summary file.

    Both the current one-row summary format and the earlier observable-indexed
    pandas format are supported.

    Returns:
        ``(models, baseline)`` where baseline is a model-key pair or ``None``.
    """
    summary = torch.load(summary_path, map_location='cpu', weights_only=False)
    if not isinstance(summary, pd.DataFrame):
        raise TypeError('The model-combination summary must contain a pandas DataFrame')

    if 'Observable' in summary.columns:
        rows = summary.loc[summary['Observable'] == observable]
        if len(rows) != 1:
            raise ValueError(
                f'Expected one {observable!r} row in {summary_path}, found {len(rows)}'
            )
        row = rows.iloc[0]
    else:
        if observable not in summary.index:
            raise KeyError(f'Observable {observable!r} is not in {summary_path}')
        row = summary.loc[observable]

    models = _normalize_models(row['Models'])
    baseline = None
    if 'Baseline Correction' in row.index and 'Baseline Model' in row.index:
        if pd.notna(row['Baseline Correction']) and pd.notna(row['Baseline Model']):
            baseline = (str(row['Baseline Correction']), str(row['Baseline Model']))
    return models, baseline


def score_single_model(model, predictions, valid_indices, sequences):
    """Calculate a model's mean Brier score over its eligible sequences."""
    scores = [
        float(
            total_brier_score(
                predictions[model]['pred_list'][index],
                sequences[index]['Label'],
            )
        )
        for index in valid_indices[model]
    ]
    if not scores or np.all(np.isnan(scores)):
        raise ValueError(f'Model {model} has no finite validation scores')
    return float(np.nanmean(scores))


def score_combined_predictions(combined_predictions, sequences):
    """Calculate the mean Brier score of index-aligned combined predictions."""
    scores = []
    for index, prediction in enumerate(combined_predictions):
        if prediction is None:
            continue
        scores.append(
            float(
                total_brier_score(
                    prediction['pred_list'],
                    sequences[index]['Label'],
                )
            )
        )
    if not scores or np.all(np.isnan(scores)):
        raise ValueError('The selected ensemble has no finite validation scores')
    return float(np.nanmean(scores))


def _parse_args():
    """Parse command-line arguments for validation inference evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions_file', type=Path, help='Running-window prediction HDF5 file')
    parser.add_argument('long_sequences', type=Path, help='Running-window sequence descriptor .pt file')
    parser.add_argument('labels_dir', type=Path, help='Directory with fixed-length sequence descriptor files')
    parser.add_argument('models_dir', type=Path, help='Directory containing transformer .pth checkpoints')
    parser.add_argument('combination_summary', type=Path, help='summary.pt from model combination')
    parser.add_argument('output_dir', type=Path, help='Directory for validation predictions and summary')
    parser.add_argument(
        '--expected-sequences',
        type=int,
        default=os.environ.get(
            'EXPECTED_SEQUENCES',
            os.environ.get('expected_sequences', 509),
        ),
    )
    parser.add_argument('--observable', choices=DEFAULT_OBSERVABLES)
    parser.add_argument('--prediction-group', help='Exact observable group name in the HDF5 file')
    parser.add_argument('--task-id', type=int, default=os.environ.get('SLURM_ARRAY_TASK_ID'))
    args = parser.parse_args()

    args.expected_sequences = int(args.expected_sequences)
    if args.expected_sequences < 1:
        parser.error('--expected-sequences must be positive')
    if args.observable is None:
        if args.task_id is None:
            parser.error('--observable or --task-id/SLURM_ARRAY_TASK_ID is required')
        try:
            args.observable = DEFAULT_OBSERVABLES[int(args.task_id)]
        except (IndexError, ValueError):
            parser.error(f'task ID must be between 0 and {len(DEFAULT_OBSERVABLES) - 1}')
    if args.prediction_group is None:
        _, args.prediction_group = convert_name_to_data_type(args.observable)
    return args


def main():
    """Evaluate the training-selected ensemble on held-out validation HARPs."""
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

    selected_models, baseline_model = load_combination_summary(
        args.combination_summary,
        args.observable,
    )
    if baseline_model is None:
        all_models = get_all_models(
            args.predictions_file,
            args.expected_sequences,
            args.prediction_group,
        )
        training_indices = get_model_validation_indices(
            all_models,
            args.labels_dir,
            args.long_sequences,
            args.models_dir,
            train=True,
        )
        baseline_correction, baseline_name, _, _ = get_baseline_performance(
            args.predictions_file,
            args.expected_sequences,
            sequences,
            training_indices,
            args.prediction_group,
        )
        baseline_model = (baseline_correction, baseline_name)

    required_models = tuple(dict.fromkeys((*selected_models, baseline_model)))
    predictions = load_selected_predictions(
        args.predictions_file,
        required_models,
        args.prediction_group,
        args.expected_sequences,
    )
    validation_indices = get_model_validation_indices(
        required_models,
        args.labels_dir,
        args.long_sequences,
        args.models_dir,
        train=False,
    )

    baseline_score = score_single_model(
        baseline_model,
        predictions,
        validation_indices,
        sequences,
    )
    combined_predictions = combine_predictions(
        selected_models,
        predictions,
        validation_indices,
        args.expected_sequences,
    )
    combined_score = score_combined_predictions(combined_predictions, sequences)
    improvement = (
        (1 - combined_score / baseline_score) * 100
        if baseline_score != 0
        else 0.0
    )

    print(f'Validation baseline {baseline_model}: {baseline_score:.6f}')
    print(f'Validation ensemble {selected_models}: {combined_score:.6f}')
    print(f'Relative improvement: {improvement:.2f}%')

    observable_dir = args.output_dir / args.observable
    observable_dir.mkdir(parents=True, exist_ok=True)
    torch.save(combined_predictions, observable_dir / 'validation_predictions.pt')
    summary = pd.DataFrame(
        [
            {
                'Observable': args.observable,
                'Prediction Group': args.prediction_group,
                'Models': selected_models,
                'Baseline Model': baseline_model,
                'Baseline Score': baseline_score,
                'Combined Score': combined_score,
                'Improvement (%)': improvement,
                'Split': 'validation',
            }
        ]
    )
    torch.save(summary, observable_dir / 'validation_summary.pt')
    csv_summary = summary.copy()
    csv_summary['Models'] = csv_summary['Models'].map(repr)
    csv_summary['Baseline Model'] = csv_summary['Baseline Model'].map(repr)
    csv_summary.to_csv(observable_dir / 'validation_summary.csv', index=False)


if __name__ == '__main__':
    main()
