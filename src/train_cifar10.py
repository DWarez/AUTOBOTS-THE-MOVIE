import sys
sys.path.append('src/modules')

import argparse

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

from vit import ViT


BATCH_SIZE = 128
EPOCHS = 1
PATH = './checkpoints/cifar_net.pth'


def create_parser():
    parser = argparse.ArgumentParser(description="Train ViT \
                                                    on CIFAR10 dataset")
    parser.add_argument("-gpu", action="store_true")
    parser.add_argument("-loadCheckpoint", action="store_true")
    return parser


def training_loop(model, trainloader, optimizer, criterion, device):
    model.to(device)
    
    print("[EPOCH ITERATION]")

    # Training loop
    for epoch in range(EPOCHS):
        running_loss = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 99 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: \
                                                    {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # Saving the model
    torch.save(model.state_dict(), PATH)


def run_testset(model, testloader, device):
    # Testset metrics
    correct = 0
    total = 0

    model.to(device)

    print("Computing accuracy on testset")

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: \
                                                    {100 * correct // total}%')



def main():
    parser = create_parser()
    args = parser.parse_args()
    device = torch.device('cuda:0' if (torch.cuda.is_available() and args.gpu) \
                                    else 'cpu')
    print(f"Using device: {device}")

    criterion = nn.CrossEntropyLoss()

    model = ViT(embedding_size=768, img_size=32, n_layers=1, n_classes=10)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    transform = transforms.Compose([transforms.ToTensor(), 
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, 
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, 
                                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, 
                                                shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, 
                                                shuffle=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 
                'horse', 'ship', 'truck')

    model = ViT(embedding_size=768, img_size=32, n_layers=1, n_classes=10)
    if args.loadCheckpoint:
        model.load_state_dict(torch.load(PATH))
        run_testset(model, testloader, device)
    else:
        training_loop(model, trainloader, optimizer, criterion, device)


if __name__ == "__main__":
    main()