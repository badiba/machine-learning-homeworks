import torch
import torchvision.transforms as T
import torch.nn.functional as F
from enum import Enum
from torch.utils.data import DataLoader
from Dataset import MnistDataset
from Model import *


def train(model, optimizer, dataloader, epochs, device):
    model.train()
    for epoch in range(epochs):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = model(images)
            loss = F.nll_loss(pred, labels)
            loss.backward()
            optimizer.step()
            # print(loss.item())


'''
def train(model, optimizer, dataloader, epochs, device):
    model.train()
    for epoch in range(epochs):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = model(images)
            loss = F.nll_loss(pred, labels)
            loss.backward()
            optimizer.step()
            # print(loss.item())
'''


def test(model, dataloader, device):
    c = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        pred = model(images)

        for i in range(pred.size()[0]):
            prediction, index = pred[i].max(0)
            if (index == labels[i]):
                c += 1

    return c


def ValidationLoss(model, dataloader, device):
    model.eval()
    c = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        pred = model(images)
        loss = F.nll_loss(pred, labels)
        loss.backward()
        print(loss.item())

    return c


def main():
    testResults = []
    useCuda = True
    device = torch.device('cuda' if useCuda else 'cpu')
    torch.manual_seed(1234)

    trainDataset = MnistDataset('data', 'train', True)
    validationDataset = MnistDataset('data', 'train', False)
    trainDataloader = DataLoader(trainDataset, batch_size=64,
                                 shuffle=False, num_workers=8)
    validationDataloader = DataLoader(validationDataset, batch_size=64,
                                      shuffle=False, num_workers=8)

    AFs = [AF.Sigmoid, AF.Tanh, AF.Relu]
    epochs = 20
    minEpoch = 5
    maxEpoch = 25

    model = ThreeLayerModel(4, 4, AF.Sigmoid, AF.Sigmoid)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    train(model, optimizer, trainDataloader, epochs, device)

    result = test(model, validationDataloader, device)
    testResults.append(result)

    for testResult in testResults:
        print(testResult)


if __name__ == '__main__':
    main()
