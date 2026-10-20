# SmartFactory: Predictive Quality Control

Coursework for **6FTC2175 Advanced Machine Learning and Neural Networks**, Assignment 1 (University of Hertfordshire). Author: Loh Zi Jian.

Builds and compares three models for predicting whether a manufactured unit will fail quality control:

| Part | Data | Models |
|---|---|---|
| A | [AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601) (sensor table) | Random Forest, feed-forward MLP |
| B | [Casting Product Image Data for Quality Inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) | CNN trained from scratch (3 versions; v3 selected in the final run) |
| C | Both | Comparison on interpretability, data and compute needs, training cost and performance; deployment recommendation |

## Headline results

| Model | Data | Test result |
|---|---|---|
| Random Forest | AI4I sensors | PR-AUC 0.902, catches 52/68 failures with 2 false alarms |
| MLP (32 → 16) | AI4I sensors | PR-AUC 0.768, catches 37/68 failures with 8 false alarms |
| CNN v3 (240k weights) | casting images | 714/715 correct (99.86%), no false alarms; optimistic, as most test photos have rotated copies in training (see Data note) |

## Running it

1. Open `AMLNN_CW1_LohZiJian.ipynb` in Google Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → Run all. Both datasets download automatically (UCI and `kagglehub`); if either is unavailable the notebook falls back to public GitHub copies and checks they match the originals. A full run takes roughly 45–55 minutes on a T4 GPU, or about 2½ hours on CPU. Set `RUN_CNN_DATA_SIZE = False` in the config cell to skip the slowest experiment.

Trained CNNs are cached in `outputs/models/` and reused on the next run. Set `SAVE_TO_DRIVE = True` to keep them on Google Drive, so a Colab disconnect doesn't lose hours of training.

To run locally instead: `pip install -r requirements.txt`, then open the notebook in Jupyter.

## Repository contents

```
AMLNN_CW1_LohZiJian.ipynb                    all code, outputs and commentary (fully executed on a Colab T4 GPU)
outputs/                                     all 14 charts from the final run (A_ = Part A, B_ = Part B, C_ = Part C)
figures/first_run_v3_batchnorm_unstable.png  earlier v3 with batch norm: unstable validation loss (discussed in B2)
figures/v3_rotation_unstable.png             earlier v3 with small-angle rotation: unstable validation accuracy (B2)
requirements.txt                             Python dependencies
```

The charts in `outputs/` are the same images shown inside the notebook. The two in `figures/` come from earlier, abandoned runs, so they are not produced by the final notebook.

## Data note

64 training images in the public casting dataset are byte-for-byte copies of test images; the notebook detects them by file hash and removes them from training. A deeper check (section B4) finds that 83% of test photos are rotated or flipped copies of parts that also appear in training, because the dataset was augmented before it was split. The CNN's test accuracy is therefore optimistic. On the 118 test photos of never-seen parts it still catches the defects, but its false-alarm rate on new good parts cannot be measured from this dataset.

## Reproducibility

All randomness is seeded (`SEED = 42`). The saved outputs come from one complete run on Google Colab (T4 GPU) with the data downloaded from UCI and Kaggle. On other hardware the neural-network results can differ slightly, because floating-point arithmetic is not bit-identical across machines; the Random Forest results are identical across runs.
