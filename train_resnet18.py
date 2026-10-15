
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "processed_images"

TRAIN_FILE = BASE_DIR / "train.txt"
VAL_FILE = BASE_DIR / "validation.txt"

NUM_CLASSES = 37
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

CHECKPOINT_PATH = BASE_DIR / "resnet18_transfer_best.pth"


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Project folder:", BASE_DIR)
print("Image folder:", IMAGE_DIR)
print("Using:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. CHECK REQUIRED PATHS
# ============================================================

for path in [IMAGE_DIR, TRAIN_FILE, VAL_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")


# ============================================================
# 4. IMAGE TRANSFORMS
# ============================================================

weights = ResNet18_Weights.DEFAULT

# ImageNet-compatible preprocessing for pretrained ResNet-18.
# Training images receive a small random augmentation.
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


# ============================================================
# 5. CUSTOM DATASET
# ============================================================

class PetDataset(Dataset):

    def __init__(self, annotation_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []

        with open(annotation_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                parts = line.rsplit(maxsplit=1)

                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid annotation line: {line}"
                    )

                filename, label = parts
                label = int(label)

                # Annotation labels in this project are 1–37.
                # PyTorch CrossEntropyLoss requires labels 0–36.
                label -= 1

                if not 0 <= label < NUM_CLASSES:
                    raise ValueError(
                        f"Label out of range in {annotation_file}: "
                        f"{filename} {label + 1}"
                    )

                self.samples.append((filename, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename, label = self.samples[index]
        image_path = self.image_dir / filename

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# 6. LOAD TRAINING AND VALIDATION DATA
# ============================================================

train_dataset = PetDataset(
    TRAIN_FILE,
    IMAGE_DIR,
    transform=train_transform
)

val_dataset = PetDataset(
    VAL_FILE,
    IMAGE_DIR,
    transform=val_transform
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError("Training or validation dataset is empty.")


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 7. LOAD PRETRAINED RESNET-18
# ============================================================

print("Loading pretrained ResNet-18...")

model = resnet18(weights=weights)

# Freeze all pretrained parameters.
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace the original 1000-class layer with 37 classes.
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(device)

print("Pretrained feature extractor: frozen")
print("New 37-class classifier: trainable")


# ============================================================
# 8. LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 9. TRAINING AND VALIDATION
# ============================================================

best_accuracy = 0.0

for epoch in range(NUM_EPOCHS):

    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = 100.0 * correct / total

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}], "
        f"Loss: {average_loss:.4f}, "
        f"Validation Accuracy: {accuracy:.2f}%"
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "accuracy": best_accuracy,
                "num_classes": NUM_CLASSES
            },
            CHECKPOINT_PATH
        )

        print("Best model saved!")

print("\nTraining completed!")
print(f"Best validation accuracy: {best_accuracy:.2f}%")
print(f"Checkpoint: {CHECKPOINT_PATH}")