import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report

from dataset import PetDataset
from model import PetResNet


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Class names
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


# Test preprocessing
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# Test dataset
test_dataset = PetDataset(
    "test.txt",
    test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# Load model
model = PetResNet().to(device)

model.load_state_dict(
    torch.load(
        "pet_resnet18.pth",
        map_location=device
    )
)

model.eval()


# Store predictions
all_labels = []
all_predictions = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(
            predictions.cpu().numpy()
        )


# Classification report
print("\nPer-breed performance:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=2
    )
)


# Confusion matrix
matrix = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:\n")

print(matrix)