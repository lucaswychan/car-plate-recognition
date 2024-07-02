import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from license_recognition.resnet import ResNet

torch.multiprocessing.set_sharing_strategy("file_system")
warnings.filterwarnings(category=UserWarning, action="ignore")


class CustomDataset(Dataset):
    def __init__(
        self, csv_file="custom_dataset.csv", image_folder="image_data", transform=None
    ):
        self.dataset_csv = pd.read_csv(csv_file)
        self.transform = transform
        self.image_folder = image_folder

        self.data = self.dataset_csv["images"].values
        self.labels = self.dataset_csv["labels"].values

        # Randomly permute the data and labels
        perm = np.random.permutation(len(self.data))
        self.data = self.data[perm]
        self.labels = self.labels[perm]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]
        image = Image.open(
            os.path.join(self.image_folder, img_name)
        )  # Convert image to RGB

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label


def load_data(batch_size, num_classes=47):

    def get_sampler(dataset):
        # solve the class imbalance problem
        count = np.zeros(num_classes)
        for _, label in dataset:
            count[label] += 1

        class_sample_count = (
            count + 0.01
        )  # add one to deal with the divided by zero case
        weight = 1.0 / class_sample_count
        samples_weight = np.array([weight[label] for _, label in dataset])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(
            samples_weight.type("torch.DoubleTensor"), len(samples_weight)
        )

        return sampler

    train_transform = transforms.Compose(
        [
            # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=3),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.5),
            transforms.ToTensor(),
            # transforms.Normalize((0.1736,0.1736,0.1736,), (0.3248,0.3248,0.3248,)),
            transforms.Normalize(
                (
                    0.1307,
                    0.1307,
                    0.1307,
                ),
                (
                    0.3081,
                    0.3081,
                    0.3081,
                ),
            ),
        ]
    )
    trnset = CustomDataset(
        csv_file="custom_dataset.csv",
        image_folder="image_data",
        transform=train_transform,
    )

    train_loader = DataLoader(
        trnset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=4,
        sampler=get_sampler(trnset),
    )

    return train_loader


def main():
    # Load the data
    batch_size = 16
    train_loader = load_data(batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize the model
    num_classes = 47
    model = ResNet(num_classes=num_classes)
    model.load_state_dict(
        torch.load("ocr.pth", map_location=device)
    )  # load the trained model in stage 1
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 200
    accuracy_train = []
    loss_train = []

    model.train()

    for epoch in range(num_epochs):
        tr_loss = 0
        all_ypred_train, all_ytrue_train = [], []

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()

            outputs = torch.nn.functional.softmax(outputs, dim=1)
            outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1)

            labels = labels.cpu().detach().numpy()

            all_ypred_train.extend(outputs)
            all_ytrue_train.extend(labels)

        loss = tr_loss / (len(train_loader))
        loss_train.append(loss)

        all_ypred_train = np.array(all_ypred_train)
        all_ytrue_train = np.array(all_ytrue_train)

        acc = accuracy_score(all_ytrue_train, all_ypred_train)
        accuracy_train.append(acc)

        print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {acc}")

    # Save the model
    torch.save(model.state_dict(), "finetuned_ocr.pth")
    np.save("accuracy_train_finetune.npy", accuracy_train)
    np.save("loss_train_finetune.npy", loss_train)


if __name__ == "__main__":
    main()
