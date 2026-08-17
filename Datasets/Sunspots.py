"""Datasets for training and applying the image-level sunspot classifier.

The training dataset pairs manually created frame labels with continuum images
stored in the project HDF5 archive. The evaluation dataset enumerates every frame
in the archive so a trained classifier can generate image-level predictions.
"""

import os

import h5py
import torchvision.transforms as transforms
import torch
import numpy as np
from tqdm.auto import tqdm


def _resolve_data_path(data_path):
    """Return an explicit HDF5 path or the ``SUNSPOT_HDF5_PATH`` fallback."""
    path = data_path or os.environ.get('SUNSPOT_HDF5_PATH')
    if path is None:
        raise ValueError(
            "Provide data_path or set the SUNSPOT_HDF5_PATH environment variable."
        )
    return path


class Manual_Labeled_Sunspot_Dataset(torch.utils.data.Dataset):
    """Pair manually labeled HMI frames with images from an HDF5 archive.

    ``labels`` maps HARP identifiers to label records. A record of
    ``(frame_index, label)`` represents one annotation. Records containing more
    than two values represent repeated annotations of the same frame; their mean
    is thresholded at 0.5. Label zero denotes no sunspot and label one denotes a
    sunspot.
    """

    def __init__(self, labels: dict, standardize=True, augmentate=True, type: str = 'continuum',
                 data_type=torch.float32, data_path=None):
        """Initialize the manually labeled image dataset.

        Args:
            labels: Existing manual-label dictionary keyed by HARP identifier.
            standardize: Scale each image to zero mean and unit variance.
            augmentate: Randomly rotate retrieved images between -45 and 45 degrees.
            type: Name of the observable dataset within each HDF5 HARP group.
            data_type: Torch dtype used for images and one-hot labels.
            data_path: HDF5 archive path. Falls back to ``SUNSPOT_HDF5_PATH``.
        """
        self.path_to_data = _resolve_data_path(data_path)
        self.type = type # continuum, magnetogram, dopplergram, ...
        self.data_type = data_type # data type of the data
        self.standardize = standardize
        if augmentate:
            # augmentate the data by rotating it
            self.transform = transforms.Compose([transforms.RandomRotation(degrees=(-45, 45)),])

        # creating look up table for labeled data of the form [(harp, index, label), ...]
        look_up = []
        harps = list(labels.keys())
        for harp in tqdm(harps, desc = "creating look_up"):
            for x in labels[harp]:
                if len(x) == 2: # in this case the images was labeled once
                    index, label = x
                    look_up.append((harp, index, label))
                else:           # in this case the images was labeled multiple times --> we take the mean
                    index, label = x[0], x[1:]
                    if np.mean(label) >= 0.5:
                        look_up.append((harp, index, 1))
                    else:
                        look_up.append((harp, index, 0))
        self.look_up = look_up

    def __len__(self):
        """Return the number of manually labeled frames."""
        return len(self.look_up)

    @staticmethod
    def standardization(data):
        """Standardize one image to zero mean and unit variance."""
        Mean = torch.mean(data)
        STD = torch.std(data)
        data = (data - Mean) / STD
        return data

    @staticmethod
    def check_for_nans(data):
        """Replace an image with zeros if it contains any NaN values."""
        if torch.isnan(data).any():
            data = torch.zeros_like(data)
        return data


    def __getitem__(self, idx): # which item do we want to take
        """Return a three-channel image, one-hot label, and dataset index."""
        harp, index, label = self.look_up[idx]
        # encode label as one-hot vector:
        if label == 0:
            label = torch.tensor([1,0], dtype = self.data_type)
        elif label == 1:
            label = torch.tensor([0,1], dtype = self.data_type)

        with h5py.File(self.path_to_data, 'r') as file:
            group = file[harp]
            data = torch.from_numpy(group[self.type][index,:,:]).to(self.data_type)

        if self.standardize:
            data = self.standardization(data)

        if hasattr(self, 'transform'):
            data = self.transform(data.unsqueeze(0)).squeeze(0)

        # check for nans:
        data = self.check_for_nans(data)

        idx = torch.tensor(idx, dtype = torch.int, requires_grad=False)
        return data.unsqueeze(0).expand(3, -1, -1), label, idx

class Manual_Labeled_Sunspot_Dataset_Evaluation(torch.utils.data.Dataset):
    """Enumerate all HDF5 frames for image-classifier inference.

    Despite the historical class name, this dataset does not require or return
    manual labels. The public ``look_up`` list maps each dataset index back to its
    ``(HARP, frame_index)`` source so predictions can be grouped by active region.
    """

    def __init__(self, standardize=True, augmentate=True, type: str = 'continuum',
                 data_type=torch.float32, data_path=None):
        """Initialize the archive-wide inference dataset.

        Args:
            standardize: Scale each image to zero mean and unit variance.
            augmentate: Randomly rotate retrieved images between -45 and 45 degrees.
            type: Name of the observable dataset within each HDF5 HARP group.
            data_type: Torch dtype used for returned images.
            data_path: HDF5 archive path. Falls back to ``SUNSPOT_HDF5_PATH``.
        """
        self.path_to_data = _resolve_data_path(data_path)
        self.type = type # continuum, magnetogram, dopplergram, ...
        self.data_type = data_type # data type of the data
        self.standardize = standardize
        if augmentate:
            # augmentate the data by rotating it
            self.transform = transforms.Compose([transforms.RandomRotation(degrees=(-45, 45)),])

        # creating look up table for labeled data of the form [(harp, index), ...]
        look_up = []
        with h5py.File(self.path_to_data, 'r') as file:
            for harp in tqdm(file.keys(), desc = "creating look_up"):
                for index in range(len(file[harp][type])):
                    look_up.append((harp, index))
        self.look_up = look_up

    def __len__(self):
        """Return the number of frames across all active regions."""
        return len(self.look_up)

    @staticmethod
    def standardization(data):
        """Standardize one image to zero mean and unit variance."""
        Mean = torch.mean(data)
        STD = torch.std(data)
        data = (data - Mean) / STD
        return data

    @staticmethod
    def check_for_nans(data):
        """Replace an image with zeros if it contains any NaN values."""
        if torch.isnan(data).any():
            data = torch.zeros_like(data)
        return data


    def __getitem__(self, idx): # which item do we want to take
        """Return a three-channel image and its archive-wide dataset index."""
        harp, index = self.look_up[idx]

        with h5py.File(self.path_to_data, 'r') as file:
            group = file[harp]
            data = torch.from_numpy(group[self.type][index,:,:]).to(self.data_type)

        if self.standardize:
            data = self.standardization(data)

        if hasattr(self, 'transform'):
            data = self.transform(data.unsqueeze(0)).squeeze(0)

        # check for nans:
        data = self.check_for_nans(data)

        idx = torch.tensor(idx, dtype = torch.int, requires_grad=False)
        return data.unsqueeze(0).expand(3, -1, -1), idx
