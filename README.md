# Visual Place Recognition Course Project (a.y. 2026-27)

> [!WARNING]
> **Student starter repository:** some parts are intentionally incomplete and
> must be implemented. The repository may also contain bugs; inspect, test, and
> debug it rather than assuming every component is correct.

## Setup

```bash
conda create -n vpr-project python=3.11
conda activate vpr-project
python -m pip install --upgrade pip
python -m pip install -e .
```

The dependencies are declared in [`pyproject.toml`](./pyproject.toml). In
Google Colab, select a GPU runtime and follow
[`Colab_examples.ipynb`](./Colab_examples.ipynb).

## Data

Run `python download_datasets.py` to download and extract the datasets under
`data/`. The default training directory is `data/gsv_xs/train`. Evaluation
expects SF-XS and Tokyo-XS test sets, and the SVOX gallery plus its five
condition-specific query folders, under `data/`.

The expected layout is:

```text
data/
├── gsv_xs/
│   └── train/
├── sf_xs/
│   └── test/
│       ├── database/
│       └── queries/
├── tokyo_xs/
│   └── test/
│       ├── database/
│       └── queries/
└── svox/
    └── images/
        └── test/
            ├── gallery/
            ├── queries_snow/
            ├── queries_rain/
            ├── queries_sun/
            ├── queries_night/
            └── queries_overcast/
```

## Student implementation tasks

The training and evaluation entry point is [`main.py`](./main.py). Before using
it, complete the marked model and loss methods:

- [`ResNetDescriptor.extract_features`](./models/resnet.py): implement the
  required ResNet truncation and return its spatial feature map.
- [`ResNetDescriptor.gem_pool`](./models/resnet.py): implement GeM pooling.
  Global average pooling is already available as the `avg` aggregation.
- [`contrastive_loss`](./utils.py): implement the contrastive objective chosen
  for the project and return a differentiable scalar loss.

These TODOs are deliberate. Training and evaluation will raise
`NotImplementedError` until the relevant methods are completed.

Once implemented, a basic training command is:

```bash
python main.py --mode train \
  --train-dir data/gsv_xs/train \
  --output-dir artifacts/my-run
```

Images are resized to 224 × 224 for both training and evaluation. During
training, a place-disjoint GSV-XS validation split is used to track recall and
select the best checkpoint. Validation samples one gallery reference and one
query image per validation place, so each query has one designated positive
gallery image by place ID. Validation R@1 selects the best checkpoint. Each
completed epoch saves `last.pt`; the best checkpoint is saved as `best.pt`.
The output directory also contains `training_history.json`.

### Resuming training

To continue a run, use the same output directory and training settings, point
`--resume` at `last.pt`, and set `--epochs` to the new total target epoch count.
For example, continue the run above to six total epochs:

```bash
python main.py --mode train \
  --train-dir data/gsv_xs/train \
  --output-dir artifacts/my-run \
  --epochs 6 \
  --resume artifacts/my-run/last.pt
```

Checkpoints contain the model, optimizer, mixed-precision scaler, and random
number generator states. Keep the output directory on persistent storage (such
as Google Drive in Colab) if you need to resume after a runtime interruption.

## Evaluation

After implementing the model and loss TODOs, evaluate the pretrained ImageNet
baseline and a trained checkpoint together:

```bash
python main.py --mode evaluate \
  --data-root data \
  --checkpoint artifacts/my-run/best.pt \
  --output-dir artifacts/my-run
```

Evaluation reports R@1, R@5, R@10, and R@20 using cosine similarity. A database
image counts as a positive when its filename-encoded UTM coordinates are
within 25 m of the query by default; adjust the radius with
`--positive-dist-threshold`. SF-XS and Tokyo-XS use their test database and
query folders. SVOX uses the gallery with only the five condition-specific
query folders listed above. The pretrained baseline and checkpoint metrics are
printed and saved to `results.json`.
