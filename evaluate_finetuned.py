
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from dataset import PetDataset


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "resnet18_finetuned_best.pth"
TEST_FILE = "test.txt"
NUM_CLASSES = 37
BATCH_SIZE = 32

print("Using device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# 2. CLASS NAMES
# --------------------------------------------------

class_names = [
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


# --------------------------------------------------
# 3. TEST TRANSFORM
# --------------------------------------------------

weights = ResNet18_Weights.DEFAULT
test_transform = weights.transforms()


# --------------------------------------------------
# 4. TEST DATASET
# --------------------------------------------------

test_dataset = PetDataset(
    TEST_FILE,
    test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_dataset))


# --------------------------------------------------
# 5. LOAD FINE-TUNED CHECKPOINT
# --------------------------------------------------

print("\nLoading fine-tuned ResNet-18...")

# Build ResNet-18 with the correct 37-class output.
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

print("Fine-tuned checkpoint loaded.")
print(
    f"Best validation accuracy: "
    f"{checkpoint['accuracy']:.2f}%"
)


# --------------------------------------------------
# 6. TEST EVALUATION
# --------------------------------------------------

all_labels = []
all_predictions = []
total_loss = 0.0
total_images = 0

criterion = nn.CrossEntropyLoss()

print("\nEvaluating test set...")

with torch.inference_mode():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        predictions = torch.argmax(outputs, dim=1)

        total_loss += loss.item() * images.size(0)
        total_images += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())


# --------------------------------------------------
# 7. RESULTS
# --------------------------------------------------

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
) * 100

test_loss = total_loss / total_images

print("\n" + "=" * 50)
print("FINE-TUNED MODEL — TEST RESULTS")
print("=" * 50)

print(f"Test images: {total_images}")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.2f}%")

print("\nPer-breed performance:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        digits=2,
        zero_division=0
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES))
    )
)
