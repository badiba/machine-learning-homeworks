import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from enum import Enum
from torch.utils.data import DataLoader
from Dataset import MnistDataset


class AF():
    Sigmoid = torch.sigmoid
    Tanh = nn.Tanh
    Relu = F.relu


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.fc1 = nn.Linear(1 * 32 * 64, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 100)

        self.functionOne = F.relu
        self.functionTwo = torch.sigmoid

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.functionOne(x)
        x = self.fc2(x)
        x = self.functionTwo(x)
        x = self.fc3(x)
        x = torch.log_softmax(x, dim=1)
        return x


class OneLayerModel(nn.Module):
    def __init__(self):
        super(OneLayerModel, self).__init__()
        self.fc1 = nn.Linear(1 * 32 * 64, 100)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.log_softmax(x, dim=1)
        return x


class TwoLayerModel(nn.Module):
    def __init__(self, iteration, activationFunction):
        super(TwoLayerModel, self).__init__()
        self.fc1 = nn.Linear(1 * 32 * 64, pow(2, 6 + iteration))
        self.fc2 = nn.Linear(pow(2, 6 + iteration), 100)
        self.activationFunction = activationFunction

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activationFunction(x)
        x = self.fc(2)
        x = torch.log_softmax(x, dim=1)


class ThreeLayerModel(nn.Module):
    def __init__(self, iterationOne, iterationTwo, functionOne, functionTwo):
        super(ThreeLayerModel, self).__init__()
        hsSizeOne = pow(2, 6 + iterationOne)
        hsSizeTwo = pow(2, 6 + iterationTwo)
        self.fc1 = nn.Linear(1 * 32 * 64, hsSizeOne)
        self.fc2 = nn.Linear(hsSizeOne, hsSizeTwo)
        self.fc3 = nn.Linear(hsSizeTwo, 100)
        self.functionOne = functionOne
        self.functionTwo = functionTwo

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.functionOne(x)
        x = self.fc2(x)
        x = self.functionTwo(x)
        x = self.fc3(x)
        x = torch.log_softmax(x, dim=1)
        return x


if __name__ == '__main__':
    dataset = MnistDataset('data', 'train', True)
    dataloader = DataLoader(dataset, batch_size=64,
                            shuffle=False, num_workers=4)
    model = BasicModel()
    for images, labels in dataloader:
        pred = model(images)
        print(pred)
        exit()
