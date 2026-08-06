"""Dataset for image-level classification of sunspots and pores."""

import os
from pathlib import Path

import torch
import numpy as np
import astropy.io.fits as fits
from scipy.optimize import curve_fit
import torchvision.transforms as transforms


class Sunspot_Dataset(torch.utils.data.Dataset):
    """Load labeled FITS images from separate ``spot`` and ``pore`` folders."""

    def __init__(self, data_dir=None, standardize=True, correct_limb_darkening=True, augmentate=True):
        """Initialize and preprocess the image-level dataset.

        Args:
            data_dir: Directory containing ``spot/`` and ``pore/`` subdirectories.
                Falls back to the ``SUNSPOT_IMAGE_DATA_DIR`` environment variable.
            standardize: Standardize each image to zero mean and unit variance.
            correct_limb_darkening: Divide each image by a fitted quadratic
                column-intensity profile.
            augmentate: Apply a random rotation whenever a sample is retrieved.

        Raises:
            ValueError: If neither ``data_dir`` nor the environment variable is set.
            FileNotFoundError: If the configured directory does not exist.
        """
        if correct_limb_darkening and not standardize:
            standardize = True
            print("If correct_limb_darkening = True, standardize will be set to True")

        self.augmentate = augmentate

        # rotate the data with an angle between -180 and 180 degrees if augmentate is True
        if augmentate:
            self.transform = transforms.Compose([transforms.RandomRotation(degrees=(-180, 180)), ])

        # Path to the data, where FITS files are stored.
        data_dir = data_dir or os.environ.get('SUNSPOT_IMAGE_DATA_DIR')
        if data_dir is None:
            raise ValueError(
                "Provide data_dir or set the SUNSPOT_IMAGE_DATA_DIR environment variable."
            )
        data_dir = Path(data_dir).expanduser()
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Sunspot image data directory not found: {data_dir}")

        # find all fits files in the spot and pore folders
        spots_fits = sorted(data_dir.glob('spot/*.fits'))
        pores_fits = sorted(data_dir.glob('pore/*.fits'))

        # create empty tensor for the data and Labels
        self.Data = torch.zeros(len(spots_fits) + len(pores_fits), 300, 300)
        # self.Labels = torch.zeros(len(spots_fits) + len(pores_fits))
        self.Labels = torch.zeros(len(spots_fits) + len(pores_fits), 2)  #  using one-hot encoding

        # fill tensors with data
        for i in range(len(spots_fits)):
            self.Data[i] = torch.tensor(fits.open(spots_fits[i])[0].data.byteswap().newbyteorder())
            # self.Labels[i] = 1
            self.Labels[i] = torch.tensor([0, 1])  #  using one-hot encoding
        for i in range(len(spots_fits), len(spots_fits) + len(pores_fits)):
            self.Data[i] = torch.tensor(fits.open(pores_fits[i - len(spots_fits)])[0].data.byteswap().newbyteorder())
            self.Labels[i] = torch.tensor([1, 0])  #  using one-hot encoding

        # fit polynomial to the average "column intensity" and divide original data by this to correct for limb darkening
        if correct_limb_darkening:
            I_column = torch.mean(self.Data, dim=1)
            fit_params = []
            fit_covs = []
            x = np.arange(I_column.shape[1])
            for r in range(I_column.shape[0]):
                p, c = curve_fit(self.f, x, I_column[r])
                fit_params.append(p)
                fit_covs.append(c)

            mean_quiet = torch.zeros((self.Data.shape[0], self.Data.shape[2]))
            for k in range(self.Data.shape[0]):
                mean_quiet[k, :] = torch.tensor(self.f(x, *fit_params[k]))
            self.Data = torch.div(self.Data, mean_quiet[:, None, :])

        # standardize the data such that each image has zero mean and unit variance
        if standardize:
            Mean = torch.mean(self.Data.view(self.Data.shape[0], -1), dim=1)
            STD = torch.std(self.Data.view(self.Data.shape[0], -1), dim=1)
            self.Data = (self.Data - Mean[:, None, None]) / STD[:, None, None]

    def __len__(self):
        """Return the number of loaded FITS images."""
        return len(self.Data)

    @staticmethod
    def f(x, a, b, c):
        """Evaluate the quadratic limb-darkening profile."""
        return a + b * x + c * x ** 2

    def __getitem__(self, idx):
        """Return one three-channel image tensor and its one-hot label."""
        data = self.Data[idx]

        if self.augmentate:
            data = self.transform(data.unsqueeze(0)).squeeze(0)

        return data.unsqueeze(0).expand(3, -1, -1), self.Labels[idx]  # .unsqueeze(0),
