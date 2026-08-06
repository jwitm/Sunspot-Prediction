"""PyTorch datasets for fixed and running-window HMI image sequences.

The datasets read active-region arrays from an HDF5 file, apply the preprocessing
used in the sunspot-evolution study, and return one-hot labels. Data locations are
provided explicitly or through environment variables so the module is portable.
"""

import os

import torch
import h5py
from tqdm import tqdm
from scipy.optimize import curve_fit
import torchvision.transforms.functional as F
import torch.nn.functional as functional


class Sequence_Dataset(torch.utils.data.Dataset):
    """Load and preprocess labeled active-region image sequences from HDF5.

    Sequence descriptors must contain ``"Harp"``, ``"Label"``, and
    ``"Sequence"`` fields. The HDF5 file must contain a group for each HARP and
    datasets named for the requested observables, plus ``bitmap`` when correction
    is enabled.
    """

    def __init__(self, sequences=None, standardize='standardize', type: str = 'continuum', data_type=torch.float32,
                 correct=True, augmentate=None, downsample=None, remove_trash=False, verbose=True,
                 data_path=None, sequences_path=None):
        """Initialize a sequence dataset.

        Args:
            sequences: In-memory sequence descriptors. If omitted, descriptors
                are loaded from ``sequences_path`` or ``SUNSPOT_SEQUENCES_PATH``.
            standardize: Scaling method: ``"standardize"``, ``"MinMax"``,
                ``"global_MinMax"``, ``"global_standardize"``, or ``None``.
            type: HDF5 observable name or list of observable names.
            data_type: Torch dtype used for returned arrays and labels.
            correct: Apply the polynomial background correction to continuum and
                Dopplergram data.
            augmentate: Iterable containing ``"flip"`` and/or ``"rotate"``.
            downsample: Integer scale factor or ``[height, width]`` output size.
            remove_trash: Remove uniform or uncorrectable frames instead of
                replacing them with zeros.
            verbose: Print dataset configuration and progress information.
            data_path: HDF5 data file. Falls back to ``SUNSPOT_HDF5_PATH``.
            sequences_path: Serialized sequence descriptors used when
                ``sequences`` is omitted. Falls back to
                ``SUNSPOT_SEQUENCES_PATH``.

        Raises:
            ValueError: If a required path or scaling method is missing or invalid.
        """
        self.correct = correct
        self.path_to_data = data_path or os.environ.get('SUNSPOT_HDF5_PATH')
        if self.path_to_data is None:
            raise ValueError(
                "Provide data_path or set the SUNSPOT_HDF5_PATH environment variable."
            )
        self.type = type  # continuum, magnetogram, dopplergram, ...
        self.data_type = data_type  # data type of the data
        self.standardize = standardize
        self.augmentate = augmentate
        self.downsample = downsample

        self.remove_trash = remove_trash

        if self.standardize not in ['standardize', 'MinMax', 'global_MinMax',
                                    'global_standardize'] and self.standardize is not None:
            raise ValueError(
                "Standardize has to be either 'standardize', 'MinMax', 'global_MinMax', 'global_standardize' or None, you want to use",
                self.standardize)

        if verbose:
            print(f"Data type: {self.data_type}")
            print(f"Type(s): {self.type}")
            print(f"Standardize: {self.standardize}")
            print(f"Correct: {self.correct}")
            print(f"Augmentate: {augmentate}")
            print(f"Downsample: {downsample}")

        if sequences is None:
            path = sequences_path or os.environ.get('SUNSPOT_SEQUENCES_PATH')
            if path is None:
                raise ValueError(
                    "Provide sequences, sequences_path, or set the "
                    "SUNSPOT_SEQUENCES_PATH environment variable."
                )
            self.sequences = torch.load(path)
            if verbose:
                print(f"Take sequences from file {path}")

        else:
            self.sequences = sequences
            if verbose:
                print("Take sequences from input")

        self.verbose = verbose
        self.max_seq_length = self.get_max_seq_length(self.sequences)

    def get_max_seq_length(self, sequences):
        """Return the longest descriptor interval in frames."""
        max_len = 0
        if self.verbose:
            iterator = tqdm(sequences, desc="Get Maximum Sequence Length")
        else:
            iterator = sequences

        for x in iterator:
            for y in x["Sequence"]:
                start, end = y
                max_len = max(max_len, end - start)
        return max_len

    @staticmethod
    def check_for_nans(data):
        """Return a copy of ``data`` with NaN values replaced by zero."""

        nan_mask = torch.isnan(data)
        data = data.clone()
        data[nan_mask] = 0

        return data

    @staticmethod
    def standardization(data, type):
        """Standardize each frame to zero mean and unit variance."""
        if type == 'continuum':
            data = torch.full_like(data, torch.max(data)) - data

        Mean = torch.mean(data, dim=(1, 2))
        STD = torch.std(data, dim=(1, 2))
        data = (data - Mean[:, None, None]) / STD[:, None, None]

        return data

    @staticmethod
    def global_standardization(data, type):
        """Standardize the complete sequence to zero mean and unit variance."""
        if type == 'continuum':
            data = torch.full_like(data, torch.max(data)) - data

        Mean = torch.mean(data)
        STD = torch.std(data)
        data = (data - Mean) / STD

        return data

    @staticmethod
    def MinMax(data, type):
        """Scale each frame independently to the interval from zero to one."""
        Min = torch.min(data.flatten(start_dim=1), dim=1)[0]
        Max = torch.max(data.flatten(start_dim=1), dim=1)[0]
        data = (data - Min[:, None, None]) / (Max - Min)[:, None, None]
        # invert continuum data
        if type == 'continuum':
            data = 1 - data
        return data

    @staticmethod
    def global_MinMax(data, type):
        """Scale all finite values in a sequence jointly to zero through one."""

        mask = ~torch.isnan(data)
        if torch.all(~mask):  # if all values are nan
            return data

        else:  # otherwise
            Min = torch.min(data[mask])
            Max = torch.max(data[mask])

            data[mask] = (data[mask] - Min) / (Max - Min)

            # invert continuum data
            if type == 'continuum':
                data[mask] = 1 - data[mask]
            return data

    @staticmethod
    def remove_sphere(x, a, b, c):
        """Evaluate the quadratic background model used for correction."""
        return a * x ** 2 + b * x + c

    def make_corrections(self, data, harp, start, end):
        """Subtract a fitted column-wise background from an image sequence.

        Args:
            data: Tensor shaped ``[time, height, width]``.
            harp: HDF5 group name for the active region.
            start: First frame index.
            end: Exclusive final frame index.

        Returns:
            The corrected tensor and a Boolean mask identifying frames for which
            no correction could be fitted.
        """
        width = torch.arange(0, data.shape[2], 1)
        # mask out active regions pixels and active pixels in general
        with h5py.File(self.path_to_data, 'r') as file:
            group = file[harp]
            bitmap = torch.from_numpy(group['bitmap'][start:end, :, :])
        masked_seq = torch.where((bitmap == 1.), data, torch.tensor(float('nan')))
        fit = []

        correction_mask = []
        for i in range(data.shape[0]):
            average_width = torch.nanmean(masked_seq[i], dim=0)
            valid_mask = ~torch.isnan(average_width)
            filtered_width = width.numpy()[valid_mask]
            filtered_average_width = average_width.numpy()[valid_mask]
            if len(filtered_average_width) != 0:
                (a, b, c), _ = curve_fit(self.remove_sphere, filtered_width, filtered_average_width)
                fit.append(torch.from_numpy(self.remove_sphere(width.numpy(), a, b, c)))
                correction_mask.append(False)
            else:
                fit.append(torch.full_like(width, torch.nan, dtype=torch.float))
                correction_mask.append(True)

        fit = torch.stack(fit)
        corrections = data - fit[:, None, :]

        return corrections, torch.tensor(correction_mask)

    @staticmethod
    def interpolate(data, downsample):
        """
        Interpolates the data to a new size.
        :param data: Input data of shape [T, H, W].
        :param downsample: New size of the data.
        :return: Interpolated data.
        """
        if type(downsample) == list:
            interpolated_data = functional.interpolate(data.unsqueeze(0), size=(downsample[0], downsample[1]),
                                                       mode='bilinear', align_corners=False).squeeze(0)
        elif type(downsample) == int:
            interpolated_data = functional.interpolate(data.unsqueeze(0),
                                                       size=(data.size(1) // downsample, data.size(2) // downsample),
                                                       mode='bilinear', align_corners=False).squeeze(0)
        return interpolated_data

    @staticmethod
    def random_rotate_sequence(sequence, angle_range=(0, 360)):
        """
        Apply the same random rotation to all frames in the sequence.
        Assumes sequence of shape [C,T, H, W].

        :param sequence: Input sequence of shape [C,T, H, W].
        :param angle_range: Range of angles for rotation (min, max).
        :return: Rotated sequence.
        """
        angle = torch.rand(1).item() * (angle_range[1] - angle_range[0]) + angle_range[0]
        rotated_sequence = torch.stack([F.rotate(frame.unsqueeze(0), angle) for frame in sequence])
        return rotated_sequence.squeeze(1)

    @staticmethod
    def is_uniform(tensor, dim1, dim2):
        """Identify slices whose values are uniform across a dimension range."""
        flattened = tensor.flatten(start_dim=dim1, end_dim=dim2)
        min_values = torch.min(flattened, dim=-1, keepdim=False).values
        max_values = torch.max(flattened, dim=-1, keepdim=False).values
        return min_values == max_values

    def __len__(self):
        """Return the number of sequence descriptors."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """Load, preprocess, and return one labeled sequence."""
        if torch.is_tensor(idx):
            idx = int(idx.item())
        elif type(idx) == float:
            idx = int(idx)
        sequence = self.sequences[idx]
        harp, label, sequence = sequence["Harp"], sequence["Label"], sequence["Sequence"]
        start, end = sequence[0]

        if type(self.type) == list:
            with h5py.File(self.path_to_data, 'r') as file:
                data = []
                for t in self.type:
                    group = file[harp]
                    sample = torch.from_numpy(group[t][start:end, :, :])

                    uniform_mask = self.is_uniform(sample, 1, 2)
                    sample[uniform_mask] = torch.nan

                    # correct for limb darkening/movements of satellite
                    if (t == 'continuum' or t == 'Dopplergram') and self.correct:
                        sample, correction_mask = self.make_corrections(sample, harp, start, end)
                        sample = sample.to(self.data_type)
                        uniform_mask = uniform_mask | correction_mask

                    if self.remove_trash:
                        sample = sample[~uniform_mask]

                    # downsample data
                    if self.downsample is not None:
                        sample = self.interpolate(sample, self.downsample)

                    # rescale data
                    if self.standardize == 'standardize':
                        sample = self.standardization(sample, self.type)
                    elif self.standardize == 'MinMax':
                        sample = self.MinMax(sample, self.type)
                    elif self.standardize == 'global_MinMax':
                        sample = self.global_MinMax(sample, self.type)
                    else:
                        pass

                    if not self.remove_trash:
                        sample[uniform_mask] = 0

                    data.append(sample)
                data = torch.stack(data)

        else:
            with h5py.File(self.path_to_data, 'r') as file:
                group = file[harp]
                data = torch.from_numpy(group[self.type][start:end, :, :])

                uniform_mask = self.is_uniform(data, 1, 2)
                data[uniform_mask] = torch.nan

            #  correct for limb darkening/movements of satellite
            if (self.type == 'continuum' or self.type == 'Dopplergram') and self.correct:
                data, correction_mask = self.make_corrections(data, harp, start, end)
                data = data.to(self.data_type)
                uniform_mask = uniform_mask | correction_mask

            if self.remove_trash:
                data = data[~uniform_mask]

            # downsample data
            if self.downsample is not None:
                data = self.interpolate(data, self.downsample)

            # rescale data
            if self.standardize == 'standardize':
                data = self.standardization(data, self.type)
            elif self.standardize == 'global_standardize':
                data = self.global_standardization(data, self.type)
            elif self.standardize == 'MinMax':
                data = self.MinMax(data, self.type)
            elif self.standardize == 'global_MinMax':
                data = self.global_MinMax(data, self.type)
            else:
                pass

            if not self.remove_trash:
                data[uniform_mask] = 0

            data = data.unsqueeze(0).expand(3, -1, -1, -1)  #  to "simulate" color channel

        # apply augmentations
        if self.augmentate is not None:
            # randomly horizontally flip all images in sequence the same way with a prop of 50%
            if 'flip' in self.augmentate and torch.randn(1) < 0.5:
                data = torch.flip(data, dims=[3])

            # randomly vertically flip all images in sequence the same way with a prop of 50%
            if 'flip' in self.augmentate and torch.randn(1) < 0.5:
                data = torch.flip(data, dims=[2])

            # randomly rotation all images in sequence the same way with a prop of 50%
            if 'rotate' in self.augmentate and torch.randn(1) < 0.5:
                data = self.random_rotate_sequence(data)

        # encode label as one-hot vector:
        if label == 0:
            label = torch.tensor([1, 0], dtype=self.data_type)
        elif label == 1:
            label = torch.tensor([0, 1], dtype=self.data_type)

        idx = torch.tensor([idx], dtype=torch.int)

        return data, label, idx


class RW_Dataset(Sequence_Dataset):
    """Expose sliding fixed-length windows from one sequence descriptor."""

    def __init__(self, sequences=None, standardize='standardize',
                 type='continuum', data_type=torch.float32, correct=True,
                 augmentate=None, downsample=None, index=0, seq_length=16, verbose=True,
                 data_path=None, sequences_path=None, remove_trash=False):
        """Initialize a running-window view of one active-region sequence."""

        # Call the parent constructor
        super().__init__(
            sequences=sequences,
            standardize=standardize,
            type=type,
            data_type=data_type,
            correct=correct,
            augmentate=augmentate,
            downsample=downsample,
            verbose=verbose,
            data_path=data_path,
            sequences_path=sequences_path,
            remove_trash=remove_trash
        )

        self.index = index
        if verbose:
            print(f"Index: {index}")
        self.sequence = self.sequences[index]
        self.seq_length = seq_length

    def __len__(self):
        """Return the number of available sliding windows."""
        _, _, sequence = self.sequence["Harp"], self.sequence["Label"], self.sequence["Sequence"]
        start, end = sequence[0]
        return end - start - self.seq_length

    def __getitem__(self, idx):
        """Load and preprocess the sliding window at ``idx``."""
        if torch.is_tensor(idx):
            idx = int(idx.item())
        elif type(idx) == float:
            idx = int(idx)
        sequence = self.sequence
        harp, label, sequence = sequence["Harp"], sequence["Label"], sequence["Sequence"]
        start, end = sequence[0]
        start = start + idx
        end = start + self.seq_length

        if type(self.type) == list:
            with h5py.File(self.path_to_data, 'r') as file:
                data = []
                for t in self.type:
                    group = file[harp]
                    sample = torch.from_numpy(group[t][start:end, :, :])

                    uniform_mask = self.is_uniform(sample, 1, 2)
                    sample[uniform_mask] = torch.nan

                    # correct for limb darkening/movements of satellite
                    if (t == 'continuum' or t == 'Dopplergram') and self.correct:
                        sample, correction_mask = self.make_corrections(sample, harp, start, end)
                        sample = sample.to(self.data_type)
                        uniform_mask = uniform_mask | correction_mask

                    if self.remove_trash:
                        sample = sample[~uniform_mask]

                    # downsample data
                    if self.downsample is not None:
                        sample = self.interpolate(sample, self.downsample)

                    # rescale data
                    if self.standardize == 'standardize':
                        sample = self.standardization(sample, self.type)
                    elif self.standardize == 'MinMax':
                        sample = self.MinMax(sample, self.type)
                    elif self.standardize == 'global_MinMax':
                        sample = self.global_MinMax(sample, self.type)
                    else:
                        pass

                    if not self.remove_trash:
                        sample[uniform_mask] = 0

                    data.append(sample)
                data = torch.stack(data)

        else:
            with h5py.File(self.path_to_data, 'r') as file:
                group = file[harp]
                data = torch.from_numpy(group[self.type][start:end, :, :])

                uniform_mask = self.is_uniform(data, 1, 2)
                data[uniform_mask] = torch.nan

            #  correct for limb darkening/movements of satellite
            if (self.type == 'continuum' or self.type == 'Dopplergram') and self.correct:
                data, correction_mask = self.make_corrections(data, harp, start, end)
                data = data.to(self.data_type)
                uniform_mask = uniform_mask | correction_mask

            if self.remove_trash:
                data = data[~uniform_mask]

            # downsample data
            if self.downsample is not None:
                data = self.interpolate(data, self.downsample)

            # rescale data
            if self.standardize == 'standardize':
                data = self.standardization(data, self.type)
            elif self.standardize == 'global_standardize':
                data = self.global_standardization(data, self.type)
            elif self.standardize == 'MinMax':
                data = self.MinMax(data, self.type)
            elif self.standardize == 'global_MinMax':
                data = self.global_MinMax(data, self.type)
            else:
                pass

            if not self.remove_trash:
                data[uniform_mask] = 0

            data = data.unsqueeze(0).expand(3, -1, -1, -1)  # to "simulate" color channel

        # apply augmentations
        if self.augmentate is not None:
            # randomly horizontally flip all images in sequence the same way with a prop of 50%
            if 'flip' in self.augmentate and torch.randn(1) < 0.5:
                data = torch.flip(data, dims=[3])

            # randomly vertically flip all images in sequence the same way with a prop of 50%
            if 'flip' in self.augmentate and torch.randn(1) < 0.5:
                data = torch.flip(data, dims=[2])

            # randomly rotation all images in sequence the same way with a prop of 50%
            if 'rotate' in self.augmentate and torch.randn(1) < 0.5:
                data = self.random_rotate_sequence(data)

        # encode label as one-hot vector:
        if label == 0:
            label = torch.tensor([1, 0], dtype=self.data_type)
        elif label == 1:
            label = torch.tensor([0, 1], dtype=self.data_type)

        idx = torch.tensor([idx], dtype=torch.int)

        return data, label, idx
