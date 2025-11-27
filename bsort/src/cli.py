"""Command Line Interface for bsort."""

import click
from src.config import Config
from src.train import train_model
from src.infer import run_inference


@click.group()
def main():
    """BSort - ML Pipeline for Object Detection."""
    pass


@main.command()
@click.option("--config", "-c", required=True, help="Path to config YAML file")
def train(config: str):
    """Train the model with given configuration.

    Args:
        config: Path to YAML configuration file
    """
    click.echo("Starting training process...")
    cfg = Config.from_yaml(config)
    train_model(cfg)


@main.command()
@click.option("--config", "-c", required=True, help="Path to config YAML file")
@click.option("--image", "-i", required=True, help="Path to image for inference")
def infer(config: str, image: str):
    """Run inference on an image.

    Args:
        config: Path to YAML configuration file
        image: Path to input image for inference
    """
    click.echo("Running inference...")
    cfg = Config.from_yaml(config)
    results = run_inference(cfg, image)

    # Print results
    for img_result in results["detections"]:
        print(f"Image {img_result['image_index'] + 1}:")
        print(f"  Detected {img_result['num_detections']} objects")
        for detection in img_result["detections"]:
            print(
                f"    Object {detection['object_id']}: "
                f"Class {detection['class']}, "
                f"Confidence: {detection['confidence']:.4f}"
            )


if __name__ == "__main__":
    main()
