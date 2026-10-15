
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image


# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "processed_images"

TRAIN_FILE = PROJECT_DIR / "train.txt"
VAL_FILE = PROJECT_DIR / "validation.txt"

TRANSFER_CHECKPOINT = PROJECT_DIR / "resnet18_transfer_best.pth"
SAVE_CHECKPOINT = PROJECT_DIR / "resnet18_finetuned_best.pth"

NUM_CLASSES = 37
BATCH_SIZE = 32
EPOCHS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Project folder: {PROJECT_DIR}")
print(f"Using: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# --------------------------------------------------
# 2. IMAGE TRANSFORMS
# --------------------------------------------------

weights = ResNet18_Weights.DEFAULT

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = weights.transforms()


# --------------------------------------------------
# 3. DATASET
# --------------------------------------------------

class PetDataset(Dataset):

    def __init__(self, annotation_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []

        with open(annotation_file, "r") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                filename, label = line.rsplit(maxsplit=1)

                image_path = self.image_dir / filename
                label = int(label) - 1

                self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# --------------------------------------------------
# 4. LOAD TRAINING AND VALIDATION DATA
# --------------------------------------------------

train_dataset = PetDataset(
    TRAIN_FILE,
    IMAGE_DIR,
    train_transform
)

val_dataset = PetDataset(
    VAL_FILE,
    IMAGE_DIR,
    val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")


# --------------------------------------------------
# 5. LOAD THE TRANSFER-LEARNING CHECKPOINT
# --------------------------------------------------

print("\nLoading ResNet-18...")

# Load the pretrained ResNet-18 architecture.
model = models.resnet18(weights=weights)

# IMPORTANT:
# The checkpoint has 37 output classes, not 1000.
# Replace the original classifier BEFORE loading it.
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

if not TRANSFER_CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {TRANSFER_CHECKPOINT}\n"
        "Run train_resnet18.py first."
    )

checkpoint = torch.load(
    TRANSFER_CHECKPOINT,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])

print("Transfer-learning checkpoint loaded.")
print(
    f"Previous best validation accuracy: "
    f"{checkpoint['accuracy']:.2f}%"
)


# --------------------------------------------------
# 6. FREEZE LAYERS AND UNFREEZE LAYER4 + CLASSIFIER
# --------------------------------------------------

# Freeze every parameter first.
for parameter in model.parameters():
    parameter.requires_grad = False

# Fine-tune the final ResNet block.
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

# Train the 37-class classifier.
for parameter in model.fc.parameters():
    parameter.requires_grad = True

model = model.to(DEVICE)

print("\nFine-tuning configuration:")
print("Frozen: conv1, bn1, layer1, layer2, layer3")
print("Trainable: layer4, fc")


# --------------------------------------------------
# 7. LOSS FUNCTION AND OPTIMIZER
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.layer4.parameters(),
            "lr": 1e-5
        },
        {
            "params": model.fc.parameters(),
            "lr": 1e-4
        }
    ],
    weight_decay=0.0001
)


# --------------------------------------------------
# 8. TRAINING AND VALIDATION
# --------------------------------------------------

best_accuracy = 0.0

print("\nStarting fine-tuning...\n")

for epoch in range(EPOCHS):

    model.train()

    # Keep the frozen layers in evaluation mode.
    # This prevents their BatchNorm statistics from changing.
    model.conv1.eval()
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_dataset)

    # Evaluate on validation data.
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    val_accuracy = 100.0 * correct / total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}], "
        f"Loss: {train_loss:.4f}, "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    # Save the best fine-tuned checkpoint.
    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "accuracy": best_accuracy,
                "num_classes": NUM_CLASSES
            },
            SAVE_CHECKPOINT
        )

        print("Best fine-tuned model saved!")


# --------------------------------------------------
# 9. TRAINING SUMMARY
# --------------------------------------------------

print("\nFine-tuning completed!")
print(f"Best fine-tuned validation accuracy: {best_accuracy:.2f}%")
print(f"Checkpoint saved at: {SAVE_CHECKPOINT}")