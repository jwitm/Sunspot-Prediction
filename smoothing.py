"""Smooth image predictions and analyze runs of binary frame labels."""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def smoothing_preds(predictions: dict, window: float = 5, verbose=True):
    """Convert class predictions to smoothed binary frame labels.

    Args:
        predictions: Mapping from HARP identifiers to tensors shaped
            ``[frames, classes]``. Column one is treated as sunspot probability.
        window: Centered moving-average width in hours at 12-minute cadence.
        verbose: Display progress information.

    Returns:
        A dictionary of binary label tensors with NaNs at window boundaries.
    """
    window_hour = window
    # convert window from hours to frames
    window_frame = int(round(window_hour * 60 / 12, 0))
    # check if window is odd, if not make it odd
    if window_frame == 1:
        window_frame_adjust = window_frame + 2
        print(
            f"window: {window_hour}h corresponds to {window_frame} frames, which is not allowed, adjusting to {window_frame_adjust} frames.")
        window = window_frame_adjust
        del (window_frame_adjust)

    elif window_frame % 2 == 0:
        window_frame_adjust = window_frame + 1
        print(
            f"window: {window_hour}h corresponds to {window_frame} frames, which is even, adjusting to {window_frame_adjust} frames.")
        window = window_frame_adjust
        del (window_frame_adjust)
    else:
        print(f"window: {window_hour}h corresponds to {window_frame} frames.")
        window = window_frame

    del (window_hour, window_frame)

    smoothing_preds = {}
    if verbose:
        for harp in tqdm(predictions, desc="Smoothing Predictions"):
            preds = predictions[harp]
            new_preds = torch.full((len(preds),), torch.nan)
            for k in range(window // 2, len(preds) - window // 2):
                average = torch.mean(preds[k - window // 2:k + window // 2, 1])
                if average >= 0.5:
                    new_preds[k] = 1
                else:
                    new_preds[k] = 0

            smoothing_preds[harp] = new_preds
    else:
        for harp in predictions:
            preds = predictions[harp]
            new_preds = torch.full((len(preds),), torch.nan)
            for k in range(window // 2, len(preds) - window // 2):
                average = torch.mean(preds[k - window // 2:k + window // 2, 1])
                if average >= 0.5:
                    new_preds[k] = 1
                else:
                    new_preds[k] = 0

            smoothing_preds[harp] = new_preds
    return smoothing_preds


def sequence_stats(labels: dict, label: int):
    """Calculate lengths of consecutive runs of a selected label.

    Args:
        labels: Mapping from HARP identifiers to arrays containing zeros and ones.
        label: Label value whose runs should be measured.

    Returns:
        A list containing the length of every matching run.
    """
    # Ensure tensor is a NumPy array
    all_lengths = []
    for harp in tqdm(labels, desc="processing active regions"):
        tensor = labels[harp]
        if not isinstance(tensor, np.ndarray):
            tensor = tensor.numpy()

        # Create a mask for the specified label
        if np.any(tensor == 0) and np.any(tensor == 1):
            is_label = tensor == label

            # Detect changes in state
            changes = np.diff(is_label.astype(int), prepend=0, append=0)
            starts = np.where(changes == 1)[0]  # Start indices of sequences
            ends = np.where(changes == -1)[0]  # End indices of sequences

            # Compute lengths of sequences
            lengths = ends - starts
            all_lengths.extend(lengths)
        elif np.any(tensor == label):
            all_lengths.extend([len(tensor)])

    return all_lengths


def count_sequences_in_window(th, window_size, labels, label):
    """Count smoothed label runs at or below a frame-length threshold."""
    sm_labels = smoothing_preds(labels, window=window_size, verbose=False)
    count = 0

    for harp in sm_labels:
        tensor = sm_labels[harp]
        if not isinstance(tensor, np.ndarray):
            tensor = tensor.numpy()

        # In this case, there are changes happening
        if np.any(tensor == 0) and np.any(tensor == 1):
            is_label = tensor == label

            # Detect changes in state
            changes = np.diff(is_label.astype(int), prepend=0, append=0)
            starts = np.where(changes == 1)[0]  # Start indices of sequences
            ends = np.where(changes == -1)[0]  # End indices of sequences

            # Compute lengths of sequences
            lengths = ends - starts
            lengths_array = np.array(lengths)

            # Count the number of sequences below the threshold
            count += np.sum(lengths_array <= th)

    return count


def count_seq_below_th(labels: dict, label: int, th: int, max_window: float = 240.4):
    """Evaluate short-run counts over a grid of smoothing-window widths.

    Args:
        labels: Prediction dictionary accepted by :func:`smoothing_preds`.
        label: Binary label whose runs should be counted.
        th: Run-length threshold in hours.
        max_window: Exclusive upper smoothing-window width in hours.

    Returns:
        Count and window-size tensors.
    """
    # Convert to frames
    th = int(round(th * 60 / 12))
    print(f"threshold: {th} frames")
    window_sizes = torch.arange(0.6, max_window, 0.4)

    counts = torch.zeros_like(window_sizes)

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(count_sequences_in_window, th, w.item(), labels, label): i for i, w in
                   enumerate(window_sizes)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Counting Sequences"):
            i = futures[future]
            counts[i] = future.result()

    return counts, window_sizes


def main():
    """Run the smoothing-window grid search for a prediction dictionary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'predictions',
        type=Path,
        help='Torch file containing a {HARP: [frames, classes]} prediction mapping',
    )
    parser.add_argument('output_dir', type=Path, help='Directory for grid-search outputs')
    parser.add_argument('--threshold-hours', type=int, default=1)
    parser.add_argument('--max-window-hours', type=float, default=51.8)
    args = parser.parse_args()

    labels = torch.load(args.predictions, map_location='cpu', weights_only=False)
    counts_neg, windows = count_seq_below_th(
        labels,
        label=0,
        th=args.threshold_hours,
        max_window=args.max_window_hours,
    )
    counts_pos, _ = count_seq_below_th(
        labels,
        label=1,
        th=args.threshold_hours,
        max_window=args.max_window_hours,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.threshold_hours}h"
    torch.save(counts_neg, args.output_dir / f"counts_neg_{suffix}.pt")
    torch.save(counts_pos, args.output_dir / f"counts_pos_{suffix}.pt")
    torch.save(windows, args.output_dir / f"windows_{suffix}.pt")
    print(f"Saved smoothing analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
