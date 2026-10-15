import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


images_folder = "processed_images"


# ImageNet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


# Training transformations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


class PetDataset(Dataset):

    def __init__(self, annotation_file, transform=None):

        self.data = []
        self.transform = transform

        with open(annotation_file, "r") as file:

            for line in file:

                parts = line.strip().split()

                image_name = parts[0]

                class_id = int(parts[1]) - 1

                image_path = os.path.join(
                    images_folder,
                    image_name
                )

                self.data.append(
                    (image_path, class_id)
                )


    def __len__(self):

        return len(self.data)


    def __getitem__(self, index):

        image_path, class_id = self.data[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return image, class_id


# Training dataset
train_dataset = PetDataset(
    "train.txt",
    train_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


# Test that the dataset works
images, labels = next(iter(train_loader))

print("Dataset loaded successfully!")
print("Number of training images:", len(train_dataset))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("First image label:", labels[0].item())