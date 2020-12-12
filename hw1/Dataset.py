import os
from enum import Enum
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


class MnistDataset(Dataset):
    def __init__(self, datasetPath, split, isTraining):
        dataFileRange = (0, 10000)
        if (isTraining):
            dataFileRange = (0, 8000)
        else:
            dataFileRange = (8000, 10000)

        imagesPath = os.path.join(datasetPath, split)
        self.data = []
        with open(os.path.join(imagesPath, 'labels.txt'), 'r') as f:
            lines = f.readlines()[dataFileRange[0]:dataFileRange[1]]
            for line in lines:
                imageName, label = line.split()
                imagePath = os.path.join(imagesPath, imageName)
                label = int(label)
                self.data.append((imagePath, label))

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, ), (0.5, )),  # mean and standard deviation
        ])

        f.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        imagePath = self.data[index][0]
        label = self.data[index][1]
        image = Image.open(imagePath)
        image = self.transforms(image)
        return image, label


if __name__ == '__main__':
    dataset = MnistDataset('data', 'train', True)
    dataloader = DataLoader(dataset, batch_size=64,
                            shuffle=False, num_workers=4)

    c = 0
    for images, labels in dataloader:
        c += labels.size()[0]

    print(c)
