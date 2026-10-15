import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import PetDataset
from model import PetResNet


# Use GPU if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Test images should only be converted and normalized
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# Load test dataset
test_dataset = PetDataset(
    "test.txt",
    test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# Create model
model = PetResNet().to(device)

# Load trained model
model.load_state_dict(
    torch.load(
        "pet_resnet18.pth",
        map_location=device
    )
)

model.eval()


# Test the model
correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)

        _, predicted = torch.max(predictions, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


accuracy = 100 * correct / total


print("Test images:", total)
print("Correct predictions:", correct)
print(f"Test Accuracy: {accuracy:.2f}%")