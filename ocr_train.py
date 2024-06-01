import warnings

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from resnet import ResNet

torch.multiprocessing.set_sharing_strategy("file_system")
warnings.filterwarnings(category=UserWarning, action="ignore")


def load_data(batch_size):
    train_transform = transforms.Compose(
        [
            # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=3),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Grayscale(num_output_channels=3),
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
    test_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
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
    trnset = torchvision.datasets.EMNIST(
        root="Data",
        split="balanced",
        train=True,
        download=True,
        transform=train_transform,
    )
    tstset = torchvision.datasets.EMNIST(
        root="Data",
        split="balanced",
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        trnset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        tstset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, test_loader


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    num_classes = 47
    model = ResNet(num_classes=num_classes).to(device)

    # Load Data
    batch_size = 64
    train_loader, test_loader = load_data(batch_size)

    # Train Model
    max_epoch = 10
    loss_train = []
    accuracy_train = []

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(max_epoch):
        print(" -- Epoch {}/{}".format(epoch + 1, max_epoch))

        running_loss = 0.0

        all_ytrue_train = []
        all_ypred_train = []

        for images, labels in tqdm(train_loader):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)

            y_pred = model(images)
            loss = criterion(y_pred, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            y_pred = torch.nn.functional.softmax(y_pred, dim=1)
            y_pred = np.argmax(y_pred.cpu().detach().numpy(), axis=1)

            labels = labels.cpu().detach().numpy()

            all_ypred_train.extend(y_pred)
            all_ytrue_train.extend(labels)

        all_ypred_train = np.array(all_ypred_train)
        all_ytrue_train = np.array(all_ytrue_train)

        loss = running_loss / (len(train_loader))
        loss_train.append(loss)
        acc = accuracy_score(all_ytrue_train, all_ypred_train)
        accuracy_train.append(acc)

        print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {acc}")

    np.save("loss_train.npy", loss_train)
    np.save("accuracy_train.npy", accuracy_train)

    # Test Model
    model.eval()
    all_ypred, all_ytrue, loss_test = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            # fetch data
            # model forward
            # calculate accuracy on test set
            images, labels = images.to(device), labels.to(device)

            y_pred = model(images)
            loss = criterion(y_pred, labels)

            loss_test.append(loss.item())

            y_pred = torch.nn.functional.softmax(y_pred, dim=1)
            y_pred = np.argmax(y_pred.cpu().detach().numpy(), axis=1)
            labels = labels.cpu().detach().numpy()

            all_ypred.extend(y_pred)
            all_ytrue.extend(labels)

    loss = np.mean(loss_test)
    acc = accuracy_score(all_ytrue, all_ypred)
    print(f"Test Stage: Loss: {loss}, Accuracy: {acc}")

    torch.save(model.state_dict(), "ocr.pth")


if __name__ == "__main__":
    main()
