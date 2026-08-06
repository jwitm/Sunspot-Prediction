# A Machine Learning Approach to Investigate the Evolution of Sunspots

This repository contains dataset preparation utilities and neural-network model definitions used to investigate whether an active region will evolve into a sunspot. The work combines convolutional neural networks (CNNs), which extract spatial information from solar images, with a Transformer encoder, which models how those features evolve over time.

The models correspond to the manuscript:

> Janis Kjell Witmer, Jonas Zbinden, Lucia Kleint, and Brandon Panos, *A Machine Learning Approach to Investigate the Evolution of Sunspots*.

## Scientific context

The study uses observations from the Helioseismic and Magnetic Imager (HMI) aboard NASA's Solar Dynamics Observatory (SDO). The analyzed observables include:

- continuum intensity images;
- line-of-sight Dopplergrams;
- line-of-sight magnetograms; and
- vector magnetic-field components.

The full methodology described in the paper has three stages:

1. A ResNet152 image classifier identifies whether an individual continuum image contains a sunspot. These predictions are temporally smoothed to generate stable labels.
2. A hybrid ResNet18-Transformer model classifies fixed-length image sequences according to whether the active region later develops a sunspot.
3. The trained sequence models are applied with a running window and combined in ensembles to produce time-resolved predictions.

The repository contains image and sequence datasets, prediction smoothing, sequence construction, and the model architectures for the first two stages. Data acquisition, training, cross-validation, ensemble selection, and evaluation code are not currently included.

## Repository structure

```text
Datasets/
├── Sequence_Dataset.py   # HDF5 sequence datasets and preprocessing
└── Sunspots.py           # FITS image-level dataset
Models/
├── Sunspot_Detection.py  # Image-level VGG/ResNet classifier
└── Transformer.py        # CNN-Transformer sequence classifier
Sequence_Builder.py       # Build sequence descriptors from frame labels
smoothing.py              # Smooth predictions and analyze label runs
```

### `Datasets/Sunspots.py`

Defines `Sunspot_Dataset`, which reads 300 x 300 FITS files from `spot/` and `pore/` directories, optionally corrects limb darkening, standardizes the images, and applies rotational augmentation.

### `Datasets/Sequence_Dataset.py`

Defines `Sequence_Dataset` for fixed HMI sequence descriptors and `RW_Dataset` for sliding-window inference. The datasets read observables from an HDF5 archive and support background correction, per-frame or sequence-level scaling, downsampling, and consistent sequence augmentation.

### `Sunspot_Detection.py`

Defines `Sunspot_CNN`, an image classifier built from an ImageNet-pretrained torchvision backbone and a fully connected two-class output head. Supported backbones are VGG16 and ResNet18/34/50/101/152. The paper uses ResNet152 for image-level sunspot classification.

### `Transformer.py`

Defines the components of the sequence-level model:

- `CNN_part` extracts a 1,000-element feature representation from each image;
- `FixedPositionalEncoding` adds sinusoidal timing information to the feature sequence; and
- `Hybrid` projects the CNN features into the Transformer embedding space, processes them with a Transformer encoder, and produces two class scores.

The `Hybrid` model expects input shaped as:

```text
[batch_size, channels, sequence_length, height, width]
```

Its output has shape `[batch_size, 2]`. The paper trains separate models for each observable rather than combining all observables as channels of one model.

### `Sequence_Builder.py`

Converts a dictionary of smoothed frame labels into positive and negative sequence descriptors. It can prune multiple intervals from the same active region, balance the classes, and exclude sequences that exceed a configurable memory threshold.

### `smoothing.py`

Applies a centered moving-average window to frame-level class predictions and provides utilities for measuring consecutive label runs and selecting a smoothing-window size.

## Requirements

- Python 3
- PyTorch
- torchvision
- NumPy
- h5py
- SciPy
- Astropy
- tqdm

Install the core dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision numpy h5py scipy astropy tqdm
```

Exact package versions were not recorded in this repository. For reproducibility, record the versions used in future training runs.

## Basic use

Run commands from the repository root so imports such as `Models.Transformer` resolve correctly.

Create an image-level classifier:

```python
from Models.Sunspot_Detection import Sunspot_CNN

model = Sunspot_CNN(base="resnet152", dropout=0.2)
```

### Trained sunspot-classifier weights

The trained ResNet152 checkpoint is not included in this GitHub repository because its approximately 242 MB file size exceeds GitHub's regular per-file limit. To use the trained classifier, obtain `Sunspot-Classification-weights.tar` separately and place it in the repository root, or change the path in the example below.

The checkpoint is a PyTorch state dictionary saved from a `DataParallel` model, so the stored parameter names begin with `module.`. Remove that prefix when loading the weights into a regular, single-device `Sunspot_CNN` instance:

```python
import torch

from Models.Sunspot_Detection import Sunspot_CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Sunspot_CNN(base="resnet152", dropout=0.2)
checkpoint = torch.load(
    "Sunspot-Classification-weights.tar",
    map_location=device,
    weights_only=True,
)
state_dict = {
    key.removeprefix("module."): value
    for key, value in checkpoint.items()
}
model.load_state_dict(state_dict)
model.to(device)
model.eval()
```

Create a sequence-level classifier:

```python
from Models.Transformer import Hybrid

model = Hybrid(
    base="resnet18",
    d_model=256,
    max_len=15,
    nhead=4,
    num_layers=3,
    dim_feedforward=1024,
    dropout=0.288,
)
```

The example sequence-model parameters reflect the architecture and optimal Transformer hyperparameters reported in the manuscript. Model construction may download pretrained torchvision weights if they are not already cached.

Load the image-level FITS dataset:

```python
from Datasets.Sunspots import Sunspot_Dataset

dataset = Sunspot_Dataset(data_dir="/path/to/SunspotPore_Data")
```

The image directory must have this structure:

```text
SunspotPore_Data/
├── pore/
│   └── *.fits
└── spot/
    └── *.fits
```

Load sequence data from HDF5:

```python
from Datasets.Sequence_Dataset import Sequence_Dataset

dataset = Sequence_Dataset(
    sequences_path="/path/to/sequences.pt",
    data_path="/path/to/hmi_active_regions.h5",
    type="continuum",
    standardize="global_MinMax",
)
```

Instead of passing paths to every dataset, they can be configured with environment variables:

```bash
export SUNSPOT_IMAGE_DATA_DIR=/path/to/SunspotPore_Data
export SUNSPOT_HDF5_PATH=/path/to/hmi_active_regions.h5
export SUNSPOT_SEQUENCES_PATH=/path/to/sequences.pt
```

Build sequence descriptors from an existing dictionary of smoothed labels:

```bash
python Sequence_Builder.py \
  /path/to/smoothed_predictions.pt \
  /path/to/output/sequences.pt \
  --corr 0 --min-length 15 --max-length 120
```

Run the smoothing-window analysis on a dictionary mapping HARP identifiers to `[frames, classes]` prediction tensors:

```bash
python smoothing.py \
  /path/to/predictions.pt \
  /path/to/output/smoothing_analysis \
  --threshold-hours 1 --max-window-hours 51.8
```

Use `python Sequence_Builder.py --help` or `python smoothing.py --help` for all command-line options.

## Data and preprocessing

The manuscript uses the HMI `hmi.sharp_cea_720s` data series, with observations binned to a 12-minute cadence. The HDF5 file expected by `Sequence_Dataset` contains one group per HARP. Each group contains datasets named after the requested observables (for example, `continuum`, `Dopplergram`, `Br`, `Bp`, and `Bt`) and a `bitmap` dataset used during background correction.

Sequence descriptor files are serialized lists of dictionaries with this structure:

```python
{
    "Harp": "HARP_GROUP_NAME",
    "Label": 0,                 # 0: no sunspot, 1: develops a sunspot
    "Sequence": [(start, end)], # end is exclusive
}
```

The implemented preprocessing includes modality-specific polynomial correction, normalization, downsampling, removal or replacement of invalid frames, rotations, and horizontal or vertical flips.

The original HMI observations are not redistributed here. Users are responsible for obtaining the data and complying with the source archive's terms and citation requirements.

## Reproducibility status

This is a partial research-code release. Dataset preprocessing and architecture definitions are available, but the repository does not yet provide a complete end-to-end reproduction package. In particular, it does not contain:

- environment lock files or pinned dependency versions;
- trained model weights, which are not distributed through this GitHub repository;
- the original observations or labeled datasets;
- image-classifier inference and prediction-cleanup code;
- training and cross-validation scripts; or
- evaluation and running-window ensemble scripts.

These limitations should be considered when reusing the models or comparing new results with the paper.

## Citation

If this code contributes to published work, please cite the associated manuscript:

```bibtex
@article{witmer2026sunspots,
  author  = {Witmer, Janis Kjell and Zbinden, Jonas and Kleint, Lucia and Panos, Brandon},
  title   = {A Machine Learning Approach to Investigate the Evolution of Sunspots},
  year    = {2026},
  note    = {Manuscript}
}
```

Replace this provisional entry with the journal citation and DOI when they are available.

## License

No license file is currently included. Until a license is added, copyright remains with the authors and reuse permissions are not explicitly granted.
