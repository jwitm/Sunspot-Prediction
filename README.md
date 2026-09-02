# A Machine Learning Approach to Investigate the Evolution of Sunspots

This repository contains dataset preparation utilities and neural-network model definitions used to investigate whether an active region will evolve into a sunspot. The work combines convolutional neural networks (CNNs), which extract spatial information from solar images, with a Transformer encoder, which models how those features evolve over time.

The models correspond to the manuscript:

> Janis Kjell Witmer, Jonas Zbinden, Lucia Kleint, and Brandon Panos, *A Machine Learning Approach to Investigate the Evolution of Sunspots*.

## Documentation notice

This README and the documentation added to the source code were generated with the assistance of artificial intelligence. The underlying research methodology, scientific results, and original implementation remain the work of the authors. Users should consult the associated manuscript and review the source code when interpreting or reusing this repository.

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

The repository contains HMI data acquisition, image and sequence datasets, prediction smoothing, sequence construction, model architectures, and running-window ensemble selection. Model-training code is not currently included.

## Repository structure

```text
Datasets/
├── Sequence_Dataset.py   # HDF5 sequence datasets and preprocessing
└── Sunspots.py           # Labeled training and archive-wide image datasets
Models/
├── Sunspot_Detection.py  # Image-level VGG/ResNet classifier
└── Transformer.py        # CNN-Transformer sequence classifier
data_pipeline.py       # Query JSOC and build the HMI HDF5 archive
Sequence_Builder.py       # Build sequence descriptors from frame labels
running_window_model_combination.py  # Select running-window ensembles
running_window_inference.py  # Evaluate selected ensembles on validation data
smoothing.py              # Smooth predictions and analyze label runs
```

### `data_pipeline.py`

Queries the definitive JSOC `hmi.sharp_cea_720s` series and can download the selected FITS segments into one HDF5 group per HARP. Its default query follows the paper: observations from 2013-11-14 through the end of 2023-04-18, with each active region's flux-weighted Stonyhurst longitude strictly between -70 and 70 degrees. It downloads the six analyzed image products (`continuum`, `magnetogram`, `Dopplergram`, `Br`, `Bp`, and `Bt`) plus `bitmap` and `conf_disambig`, which support masking and magnetic-field preprocessing.

### `Datasets/Sunspots.py`

Defines the two image-level datasets used in the labeling workflow:

- `Manual_Labeled_Sunspot_Dataset` combines an existing dictionary of manual frame labels with images in the HMI HDF5 archive. Repeated annotations of a frame are averaged and thresholded at 0.5. It returns standardized, optionally rotated, three-channel images together with one-hot labels.
- `Manual_Labeled_Sunspot_Dataset_Evaluation` enumerates every frame of a selected observable in the HDF5 archive for classifier inference. Despite its historical name, this class does not require or return manual labels. Its `look_up` attribute maps dataset indices back to HARP and frame indices.

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

### `running_window_model_combination.py`

Evaluates running-window prediction models with the Brier score and searches non-empty model subsets for the best ensemble. For each model, it reconstructs the original five-fold split from the checkpoint's random seed and includes only active regions belonging to the selected training or validation split. Ensemble probabilities are averaged, while predictive uncertainty combines the mean within-model variance with the variance between model predictions.

The script operates on predictions that have already been generated. It does not load the Transformer architectures or perform running-window model inference itself.

### `running_window_inference.py`

Loads the model subset and baseline selected by `running_window_model_combination.py`, reconstructs their held-out cross-validation folds, and evaluates the ensemble on validation HARPs. It saves index-aligned combined predictions and validation summaries in both PyTorch and CSV formats.

Despite its filename, this script combines previously generated model outputs rather than executing the Transformer models. The running-window predictions must already exist in the HDF5 archive.

## Requirements

- Python 3
- PyTorch
- torchvision
- NumPy
- h5py
- drms
- Astropy
- pandas
- SciPy
- scikit-learn
- tqdm

Install the core dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision numpy h5py drms astropy pandas scipy scikit-learn tqdm
```

Exact package versions were not recorded in this repository. For reproducibility, record the versions used in future training runs.

## Basic use

Run commands from the repository root so imports such as `Models.Transformer` resolve correctly.

Preview the paper-matched JSOC selection and its estimated size:

```bash
python data_pipeline.py
```

The generated DRMS record-set query is:

```text
hmi.sharp_cea_720s[][2013.11.14_00:00:00_TAI-2023.04.18_23:59:59_TAI][? LON_FWT > -70 AND LON_FWT < 70 ?]
```

The empty first selector includes every HARP, the second selector covers the paper's complete date interval, and the final selector applies the strict longitude bounds to each observation. `LON_FWT` is the Stonyhurst longitude of the line-of-sight flux-weighted center of the active patch.

Downloading is deliberately opt-in because the manuscript reports a total size of approximately 7 TB:

```bash
python data_pipeline.py \
  --download \
  --output /path/to/hmi_active_regions.h5 \
  --max-workers 8
```

The output parent directory is created automatically. Querying and downloading require network access to JSOC; an interrupted multi-terabyte download is not currently resumable.

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

### Image-level labeling datasets

The manually labeled data must already exist before constructing the training dataset. The repository does not create these annotations or include the label dictionary. In the paper, 1,000 continuum images from 595 active regions were manually labeled according to whether they contained a sunspot.

The expected dictionary maps each HARP identifier to label records:

```python
manual_labels = {
    "3386": [
        (12, 0),          # frame 12 was labeled once as no sunspot
        (45, 1),          # frame 45 was labeled once as sunspot
        (78, 1, 1, 0),    # repeated labels are averaged and thresholded at 0.5
    ],
}
```

The HARP keys must match groups in the HDF5 archive, and each frame index must be valid for the selected observable. Construct the labeled training dataset as follows:

```python
import torch

from Datasets.Sunspots import Manual_Labeled_Sunspot_Dataset

manual_labels = torch.load(
    "/path/to/manual_labels.pt",
    map_location="cpu",
    weights_only=False,
)
dataset = Manual_Labeled_Sunspot_Dataset(
    labels=manual_labels,
    data_path="/path/to/hmi_active_regions.h5",
    type="continuum",
    standardize=True,
    augmentate=True,
)
```

Train/validation splitting is not performed by the dataset and must be applied separately. The paper uses 90% of the manually labeled images for training and 10% for validation. Training images are standardized to zero mean and unit variance and randomly rotated between -45 and 45 degrees; validation data should normally be constructed with `augmentate=False`.

After training, use the evaluation dataset to enumerate all images in the HDF5 archive and generate model-based labels:

```python
from Datasets.Sunspots import Manual_Labeled_Sunspot_Dataset_Evaluation

inference_dataset = Manual_Labeled_Sunspot_Dataset_Evaluation(
    data_path="/path/to/hmi_active_regions.h5",
    type="continuum",
    standardize=True,
    augmentate=False,
)

# Map a returned dataset index back to its source HARP and frame.
harp, frame_index = inference_dataset.look_up[0]
```

This second dataset does not require the manual-label dictionary. It requires only the prepared HDF5 archive and is intended for applying the trained classifier to the remaining images.

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

Select a running-window ensemble on the training folds:

```bash
python running_window_model_combination.py \
  /path/to/running_window_predictions.h5 \
  /path/to/long_sequences.pt \
  /path/to/fixed_sequence_labels \
  /path/to/transformer_checkpoints \
  /path/to/output \
  --expected-sequences 509 \
  --observable Continuum \
  --prediction-group continuum \
  --processes 8
```

The five positional paths are, in order:

1. the HDF5 file containing previously generated running-window predictions;
2. the running-window sequence descriptor file used as the reference;
3. the directory containing fixed-length sequence files for each correction value;
4. the directory tree containing the trained Transformer checkpoints; and
5. the output directory for combined predictions and score summaries.

`--prediction-group` must exactly match the observable group name in the prediction HDF5 file. If it is omitted, the value supplied through `--observable` is used. By default, model eligibility and scores are calculated from training HARPs; pass `--validation` to use the corresponding validation HARPs. The script also accepts `SLURM_ARRAY_TASK_ID`, `SLURM_CPUS_PER_TASK`, `EXPECTED_SEQUENCES`, and the legacy lowercase `expected_sequences` environment variable, but none of them is required when the equivalent command-line options are provided.

The subset search grows exponentially with the number of candidate models. The paper evaluates all non-empty combinations of a preselected pool of 19 models per observable. The supplied HDF5 file should therefore contain only the intended candidate pool, and `--max-subset-size` can be used to limit the largest ensemble considered.

Evaluate the selected ensemble on the held-out validation folds:

```bash
python running_window_inference.py \
  /path/to/running_window_predictions.h5 \
  /path/to/long_sequences.pt \
  /path/to/fixed_sequence_labels \
  /path/to/transformer_checkpoints \
  /path/to/model_combination_output/Continuum/summary.pt \
  /path/to/validation_output \
  --expected-sequences 509 \
  --observable Continuum \
  --prediction-group continuum
```

The inference script reads the selected model tuple and training-selected baseline from the combination summary. It calculates both scores using only each model's validation HARPs, saves `validation_predictions.pt`, and writes `validation_summary.pt` and `validation_summary.csv`. Current summaries contain the baseline identity directly; summaries created by older versions are also supported, although finding the baseline then requires recalculating training scores.

Use each script's `--help` option for its complete command-line reference.

## Data and preprocessing

The manuscript uses the HMI `hmi.sharp_cea_720s` data series, whose records have a 12-minute cadence. The default query in `data_pipeline.py` implements the paper's time and longitude criteria. The resulting HDF5 file contains one group per HARP, with datasets named after the requested observables (`continuum`, `magnetogram`, `Dopplergram`, `Br`, `Bp`, and `Bt`) and auxiliary `bitmap` and `conf_disambig` masks.

Sequence descriptor files are serialized lists of dictionaries with this structure:

```python
{
    "Harp": "HARP_GROUP_NAME",
    "Label": 0,                 # 0: no sunspot, 1: develops a sunspot
    "Sequence": [(start, end)], # end is exclusive
}
```

The implemented preprocessing includes modality-specific polynomial correction, normalization, downsampling, removal or replacement of invalid frames, rotations, and horizontal or vertical flips.

The running-window prediction HDF5 file must use the following hierarchy:

```text
observable/
└── correction/
    └── model_name/
        └── sequence_index/
            ├── predictions
            └── uncertainty
```

The last column of each `predictions` and `uncertainty` array is interpreted as the sunspot-class probability or uncertainty. Fixed-length sequence descriptor filenames must follow `Sequences_ws_23p8_corr_<correction>_filter_[15, 15].pt`. Each model filename must begin with its fold number and end in `.pth`; its checkpoint must contain `parameters.random_state` so the original five-fold split can be reconstructed.

The original HMI observations are not redistributed here. Users are responsible for obtaining the data and complying with the source archive's terms and citation requirements.

## Reproducibility status

This is a partial research-code release. Dataset preprocessing and architecture definitions are available, but the repository does not yet provide a complete end-to-end reproduction package. In particular, it does not contain:

- environment lock files or pinned dependency versions;
- trained model weights, which are not distributed through this GitHub repository;
- the original observations or labeled datasets;
- image-classifier inference and prediction-cleanup code;
- training scripts; or
- the code that initially generates the running-window prediction HDF5 archive.

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

## License

No license file is currently included. Until a license is added, copyright remains with the authors and reuse permissions are not explicitly granted.
