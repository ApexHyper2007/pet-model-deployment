import argparse
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models
from torchvision.models import ResNet18_Weights


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_DIR / "resnet18_finetuned_best.pth"
NUM_CLASSES = 37

# These names must match the 1-based Oxford-IIIT Pet labels converted to 0-based indices.
CLASS_NAMES = [
    "Abyssinian",
    "American Bulldog",
    "American Pit Bull Terrier",
    "Basset Hound",
    "Beagle",
    "Bengal",
    "Birman",
    "Bombay",
    "Boxer",
    "British Shorthair",
    "Chihuahua",
    "Egyptian Mau",
    "English Cocker Spaniel",
    "English Setter",
    "German Shorthaired",
    "Great Pyrenees",
    "Havanese",
    "Japanese Chin",
    "Keeshond",
    "Leonberger",
    "Maine Coon",
    "Miniature Pinscher",
    "Newfoundland",
    "Persian",
    "Pomeranian",
    "Pug",
    "Ragdoll",
    "Russian Blue",
    "Saint Bernard",
    "Samoyed",
    "Scottish Terrier",
    "Shiba Inu",
    "Siamese",
    "Sphynx",
    "Staffordshire Bull Terrier",
    "Wheaten Terrier",
    "Yorkshire Terrier"
]


def load_model(device):
    """Rebuild and load the fine-tuned 37-class ResNet-18."""
    # The checkpoint contains weights only, so recreate the model architecture first.
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    # Use no ImageNet weights here because the fine-tuned checkpoint supplies the weights.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError("The checkpoint does not contain 'model_state_dict'.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_pil_image(model, image, device, top_k=5):
    """Return the top predictions for one PIL image."""
    # Match the validation/test preprocessing used during model evaluation.
    transform = ResNet18_Weights.DEFAULT.transforms()
    image_tensor = transform(image.convert("RGB"))

    # Models expect a batch dimension, even when predicting one image.
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # inference_mode disables gradients and reduces unnecessary inference overhead.
    with torch.inference_mode():
        # Softmax converts model logits into probabilities for the displayed scores.
        probabilities = torch.softmax(model(image_tensor), dim=1)
        top_probabilities, top_indices = torch.topk(
            probabilities, k=min(top_k, len(CLASS_NAMES)), dim=1
        )

    return [
        (CLASS_NAMES[index.item()], probability.item())
        for probability, index in zip(
            top_probabilities[0].cpu(), top_indices[0].cpu()
        )
    ]


def predict_image(model, image_path, device, top_k=5):
    """Open an image file and return its top predictions."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(image_path) as image:
            return predict_pil_image(model, image, device, top_k)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            f"Could not read image '{image_path}'. Check that it is a valid image file."
        ) from error


def predict_image_bytes(model, image_bytes, device, top_k=5):
    """Decode image bytes and return their top predictions."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            return predict_pil_image(model, image, device, top_k)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            "Could not read the uploaded file. Check that it is a valid image."
        ) from error


def main():
    # Read the image path from the command line.
    parser = argparse.ArgumentParser(
        description="Predict the breed in a pet image."
    )
    parser.add_argument("image", type=Path, help="Path to a pet image")
    args = parser.parse_args()

    # Prefer the NVIDIA GPU when CUDA is available; otherwise use the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model = load_model(device)
        predictions = predict_image(model, args.image, device)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(f"\nPredicted breed: {predictions[0][0]}")
    print(f"Confidence: {predictions[0][1] * 100:.2f}%")
    print("\nTop 5 predictions:")

    for rank, (breed, probability) in enumerate(predictions, start=1):
        print(f"{rank}. {breed}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()
