"""Build labeled HMI sequence descriptors from smoothed image predictions."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm


def build_sequences(labels: dict, corr: int = 0, filter: list = [0, 200], prune=True, balance=False):
    """Build sequence descriptors from a dictionary of smoothed labels.

    Sequences are built as follows:
    - if there is a change from 0 to 1, the sequence starts
    - if there is a change from 1 to 0, the sequence ends
    - if there is no change in the label and the label is 0, the sequence is the whole interval
    - if there is no change in the label and the label is 1, the sequence is not considered

    Args:
        labels: Mapping from HARP identifiers to smoothed frame labels.
        corr: Offset applied to the positive-sequence end. Positive values stop
            before formation; negative values include sunspot frames.
        filter: Minimum and maximum sequence lengths in frames.
        prune: Keep only the longest valid sequence per active region.
        balance: Randomly downsample the majority class.

    Returns:
        Dictionaries containing ``Harp``, ``Label``, and ``Sequence`` fields.
    """

    sequences = []
    for harp in tqdm(labels, desc="processing active regions"):
        tensor = labels[harp]
        if not isinstance(tensor, np.ndarray):
            tensor = tensor.cpu().numpy()

        # there will be a chage in label
        if np.any(tensor == 0) and np.any(tensor == 1):

            # check where there are 0 and assign 1
            is_label = (tensor == 0).astype(int)

            is_nan = np.isnan(tensor)
            # assign 3 to nan values and for the rest take the is_label mask
            is_label_with_nan = np.where(is_nan, 3, is_label)

            #  detect changes in the label
            """
            nan --> spot: -3
            nan --> pore: -2
            pore --> spot: -1
            no-change: 0
            spot --> pore: 1
            pore --> nan: 2
            spot --> nan: 3

            relevant for me are (-2,1) as starts and -1 as end
            """
            changes = np.diff(is_label_with_nan.astype(int), prepend=0, append=0)

            starts = np.where((changes == -2) | (changes == 1))[0]
            ends = np.where(changes == -1)[0]

            #  where spots end
            ends_sp = np.where((changes == 3) | (changes == 1))[0]

            # if there is no end, the sequence is still ongoing make correction
            if len(ends) > 0:

                # there can't be starts after the last end
                starts = starts[starts <= np.max(ends)]
                ends_sp = ends_sp[ends_sp >= np.min(ends)]
                seq = []
                for start, end, end_spot in zip(starts, ends, ends_sp):
                    #  apply the filter

                    # NOTE: this change I have recently made to the code (end --> end-corr) in the line below
                    # NOTE: in the last line (seq.append) the end-corr was already there...
                    if end - corr - start >= filter[0] and end_spot - end >= filter[0]:
                        if (end - corr - start) > filter[1]:
                            correction = ((end - corr - start) - filter[1])
                            start += correction
                        seq.append((start, end - corr))
                if len(seq) > 0:
                    sequences.append({"Harp": harp, "Label": 1, "Sequence": seq})
            # if end is empty, the is no sequence
            else:
                pass

        # there will be no change in label and the label is 0
        elif np.any(tensor == 0):
            check_where_not_nans = np.where(tensor == 0)[0]
            start, end = check_where_not_nans[0], check_where_not_nans[-1]
            if (end - start) >= filter[0]:
                if (end - start) > filter[1]:
                    total_reduction = (end - start) - filter[1]
                    start_correction = total_reduction // 2
                    end_correction = total_reduction - start_correction
                    start += start_correction
                    end -= end_correction

                #    correction = ((end-start)-filter[1])//2
                #    start+=correction
                #    end-=correction
                sequences.append({"Harp": harp, "Label": 0, "Sequence": [(start, end)]})

    if prune:
        "this will ensure, that only the longest sequence is taken from each active region and therefore the train- test split will not be biased by the same active region"
        for x in sequences:
            if len(x["Sequence"]) > 1:
                max_seq_length = 0
                longest_seq = 0
                for y in x["Sequence"]:
                    seq_len = y[1] - y[0]
                    seq = y
                    if seq_len > max_seq_length:
                        max_seq_length = seq_len
                        longest_seq = seq
                x["Sequence"] = [longest_seq]

    if balance:
        positives = [idx for idx in range(len(sequences)) if sequences[idx]["Label"] == 1]
        negatives = [idx for idx in range(len(sequences)) if sequences[idx]["Label"] == 0]

        if len(positives) > len(negatives):
            valid_positives = np.random.choice(positives, len(negatives), replace=False)
            negatives.extend(valid_positives)
            indices = negatives
            print(f"Removing {len(positives) - len(negatives)} positive sequences")
        elif len(negatives) > len(positives):
            valid_negatives = np.random.choice(negatives, len(positives), replace=False)
            positives.extend(valid_negatives)
            indices = positives
            print(f"Removing {len(negatives) - len(positives)} negative sequences")
        else:
            indices = np.arange(len(sequences))
            print(f"Number of positive and negative sequences is balanced")
    else:
        indices = np.arange(len(sequences))

    valid_sequences = [sequences[i] for i in indices]

    return valid_sequences


def check_if_model_runs(model, loader, threshhold=0.9):
    """Return sizes and shapes of batches that fail or use too much GPU memory."""
    problematic_numbers = []
    problematic_shapes = []

    simulating_output = []

    total_memory = torch.cuda.get_device_properties(0).total_memory

    for data, label, idx, mask in tqdm(loader, desc="feed data to model"):
        try:
            data = data.to('cuda')
            label = label.to('cuda')
            mask = mask.to('cuda')
            out = model(data, mask)
            simulating_output.append(out.detach().cpu())
            current_memory = torch.cuda.memory_allocated()
            if current_memory / total_memory > threshhold:
                problematic_numbers.append(np.prod(data.shape))
                problematic_shapes.append(data.shape)
            del out
        except Exception as e:
            problematic_numbers.append(np.prod(data.shape))
            problematic_shapes.append(data.shape)
            print(e)
        del data, label, mask
        torch.cuda.empty_cache()
    return problematic_numbers, problematic_shapes


def filter_sequences(sequences: list, max_floats: float = 7.34e8, data_path=None):
    """Remove sequences whose continuum arrays exceed ``max_floats`` elements."""
    import h5py

    if type(max_floats) == float:
        max_floats = int(max_floats)

    path_to_data = data_path or os.environ.get('SUNSPOT_HDF5_PATH')
    if path_to_data is None:
        raise ValueError(
            "Provide data_path or set the SUNSPOT_HDF5_PATH environment variable."
        )

    indices = []

    with h5py.File(path_to_data, 'r') as file:
        for i, x in enumerate(tqdm(sequences, desc="Filtering Sequences")):
            harp = x["Harp"]
            start, end = x["Sequence"][0]
            group = file[harp]
            data = group["continuum"][start:end, :, :]
            if np.prod(data.shape) >= max_floats:
                indices.append(i)
                print(f"Harp Number {harp} will be removed due to memory constraints (>{max_floats})")
    sequences = [x for i, x in enumerate(sequences) if i not in indices]
    return sequences


def main():
    """Build and save sequences from a serialized dictionary of frame labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('labels', type=Path, help='Torch file containing smoothed labels by HARP')
    parser.add_argument('output', type=Path, help='Destination .pt file for sequence descriptors')
    parser.add_argument('--corr', type=int, default=0, help='End offset in frames (default: 0)')
    parser.add_argument('--min-length', type=int, default=0, help='Minimum sequence length in frames')
    parser.add_argument('--max-length', type=int, default=200, help='Maximum sequence length in frames')
    parser.add_argument('--balance', action='store_true', help='Downsample the majority class')
    parser.add_argument('--no-prune', action='store_true', help='Keep multiple sequences per active region')
    parser.add_argument('--data-path', type=Path, help='HDF5 file used with --max-floats')
    parser.add_argument('--max-floats', type=float, help='Remove sequences at or above this element count')
    args = parser.parse_args()

    labels = torch.load(args.labels, map_location='cpu', weights_only=False)
    sequences = build_sequences(
        labels,
        corr=args.corr,
        filter=[args.min_length, args.max_length],
        prune=not args.no_prune,
        balance=args.balance,
    )
    if args.max_floats is not None:
        sequences = filter_sequences(
            sequences,
            max_floats=args.max_floats,
            data_path=args.data_path,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sequences, args.output)
    print(f"Saved {len(sequences)} sequences to {args.output}")


if __name__ == "__main__":
    main()
