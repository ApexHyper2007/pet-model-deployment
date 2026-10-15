import torch
import torch.nn as nn
import torch.optim as optim

from dataset import PetDataset, train_loader
from model import PetResNet
from torch.utils.data import DataLoader
from torchvision import transforms


# Use GPU if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------
# Validation dataset
# --------------------

validation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

validation_dataset = PetDataset(
    "validation.txt",
    validation_transform
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False
)


# --------------------
# Create model
# --------------------

model = PetResNet().to(device)


# --------------------
# Loss function
# --------------------

loss_function = nn.CrossEntropyLoss()


# --------------------
# Optimizer
# --------------------

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)


# Number of training rounds
epochs = 5

# Keep track of best validation accuracy
best_accuracy = 0


# --------------------
# Training
# --------------------

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        # Move data to GPU
        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Make predictions
        predictions = model(images)

        # Calculate loss
        loss = loss_function(predictions, labels)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        total_loss += loss.item()


    average_loss = total_loss / len(train_loader)


    # --------------------
    # Validation
    # --------------------

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            # Move data to GPU
            images = images.to(device)
            labels = labels.to(device)

            # Make predictions
            predictions = model(images)

            # Get predicted class
            _, predicted = torch.max(predictions, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()


    accuracy = 100 * correct / total


    # --------------------
    # Display results
    # --------------------

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {average_loss:.4f}, "
        f"Validation Accuracy: {accuracy:.2f}%"
    )


    # --------------------
    # Save best model
    # --------------------

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(
            model.state_dict(),
            "best_pet_resnet18.pth"
        )

        print("Best model saved!")


print("Training completed!")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")