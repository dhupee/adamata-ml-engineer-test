# Adamata ML Engineer Test

This is my submission for ML Engineer Test by Adamata

<!--toc:start-->
- [Adamata ML Engineer Test](#adamata-ml-engineer-test)
  - [Important Note for Adamata Team](#important-note-for-adamata-team)
  - [How to use this project](#how-to-use-this-project)
    - [Notebook](#notebook)
    - [Bsort Program](#bsort-program)
  - [What I did with the Dataset](#what-i-did-with-the-dataset)
  - [Model Performance](#model-performance)
  - [Model Performance Summary](#model-performance-summary)
    - [Validation Performance](#validation-performance)
    - [Per-Class Performance](#per-class-performance)
    - [Inference Speed](#inference-speed)
  - [What would I do differently](#what-would-i-do-differently)
<!--toc:end-->

## Important Note for Adamata Team

Given few circumstance I had this week, I made few changes and deviation from the requirements

1. I did not include Dockerfile, Github Action due to the time constraint, and I lost a full day because I had a Graduation, but I replace them by using UV to provides similar level of reproducability both in dev and prod.
2. I don't include wandb, for some reason wandb refuses to `create project`, and the forum can't understand the issue because the error message was so vague no one could narrow it down.
3. I only use Yolo11n for this test, given enough days I might test another models.

## How to use this project

### Notebook

for notebook, please use [Kaggle](https://www.kaggle.com/) by import the notebook, many of the directories, and functions are design with Kaggle in mind.

### Bsort Program

This program is written to utilize [UV](https://docs.astral.sh/uv/), Please install it as per guide in the site,
then on `bsort` directory simply run

```bash
uv run bsort train --config settings.yaml
```

or

```bash
uv run bsort infer --config settings.yaml --image <image_path>
```

That command will install

- Install Python versions it needed and create a virtual environment
- Install Library it needs
- Runs the CLI

You can however not use UV and simply use Python 3.11 then install the library yourself since the `pyproject.toml` provides the data on what the project needs, however reproducability isnt guaranteed.

## What I did with the Dataset

Basically because we only got 12 Images, and the label is all wonky I decided to use [Roboflow](https://roboflow.com/) to annotate, but also augmented it.

From given dataset I did this

1. Preprocess the image by squash them to 320x320 image
2. Flip: Horizontal, Vertical
3. Crop: 0% Minimum Zoom, 15% Maximum Zoom
4. Rotation: Between -10° and +10°
5. Brightness: Between -15% and +15%
6. Blur: Up to 1.2px
7. Noise: Up to 0.1% of pixels

That give me extra 16 images, so I get 28.

Roboflow also helps me to format the dataset into other format like COCO, which should be helpful in the future.

## Model Performance

Here's a markdown-friendly version of the YOLO11n performance summary that you can add directly to your README:

## Model Performance Summary

Model I used is Yolo11n, and from its Validation I got

### Validation Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **mAP50** | 99.5% | Primary accuracy metric (IoU ≥ 50%) |
| **mAP50-95** | 85.7% | Strict accuracy metric (IoU 50%-95%) |
| **Precision** | 91.6% | Correct detections (low false positives) |
| **Recall** | 100% | Objects found (no missed detections) |

### Per-Class Performance

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| **dark-blue** | 2 | 11 | 99.1% | 100% | 99.5% | 77.7% |
| **others** | 1 | 7 | 84.1% | 100% | 99.5% | 93.7% |
| **All** | 3 | 18 | 91.6% | 100% | 99.5% | 85.7% |

### Inference Speed

| Stage | Time |
|-------|------|
| Preprocess | 0.7ms |
| **Inference** | **45.7ms** |
| Postprocess | 3.0ms |
| **Total FPS** | **~22 FPS** |

*Note: Test conducted on 3 images containing 18 total object instances.*

## What would I do differently

Given enough time, I might:

1. Playing with other than YOLO, smaller model perhaps that usually aimed for Microcontroller.
2. Actually testing it on VM using QEMU.
