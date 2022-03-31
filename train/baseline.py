import os 
import torch
import argparse
import os
import sys
from pprint import pprint
from torchvision import models, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pdb 

class SceneRecognitionDataLoader(Dataset):
    def __init__(self, file, task, root, transform=None):
        self.file = open(file)
        self.transform = transform
        self.root = root
        self.imageFiles = []

        for line in self.file:
            ID, imgFile = line.split(" ")
            ID = int(ID)

            imgFile = imgFile.strip("\n")
            if task == "openset":
                if ID < 50:
                    pass 

                elif ID >= 50 and ID < 75:
                    ID = 50

                else:
                    continue 

            self.imageFiles.append((ID, imgFile))

    def __len__(self):
        return len(self.imageFiles)

    def __getitem__(self, index):
        ID, imgFile = self.imageFiles[index]
        img = Image.open(self.root + "/" + imgFile).convert("RGB")

        if self.transform:
            img = self.transform(img)

        ID = torch.tensor(ID)

        return img, ID

class ResNet(nn.Module):
    def __init__(self, numClasses, pretrain=False):
        super(ResNet, self).__init__()
        self.feats = nn.Sequential(*list(models.resnet18(pretrained=pretrain).children())[:-1])
        self.classifier = nn.Sequential(nn.Linear(512, numClasses))
    
    def forward(self, img):
        feats = self.feats(img).reshape(-1, 512)
        preds = self.classifier(feats)

        return preds


def saveCheckPoint(model, device, modelState):
    fileName = "../models/baseline/" + modelState["experimentName"] + "/modelCheckPoint" + str(modelState["Epoch"]) + ".pth.tar"
    torch.save(modelState, fileName)


def train(args):
    experiment = args.experiment
    numClasses = args.numClasses
    dataFile = args.dataFile
    root = args.root
    BATCHSIZE = args.batchSize
    EPOCH = args.numEpoch
    WORKERS = args.workers
    LR = args.learningRate
    WEIGHTDECAY = args.weightDecay
    pretrain = args.pretrain

    if not os.path.isdir("../models/baseline/" + experiment + "/"):
        os.makedirs("../models/baseline/" + experiment + "/")

    print("Using Parameters:")
    pprint(vars(args))

    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TrainDataset = SceneRecognitionDataLoader(dataFile, experiment, root, transform)
    trainData = DataLoader(TrainDataset, batch_size=BATCHSIZE, shuffle=True, num_workers=WORKERS)

    model = ResNet(numClasses, pretrain)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999), weight_decay=WEIGHTDECAY)
    cross_entropy = nn.CrossEntropyLoss()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        optimizer = optim.Adam(model.module.parameters(), lr=LR, betas=(0.5, 0.999), weight_decay=WEIGHTDECAY)

    model.to(device)

    for STEP in range(EPOCH):
        epochLoss = 0
        model.train()

        for data in trainData:
            img, target = data

            img = img.to(device)
            target = target.to(device)

            pred = model(img)

            optimizer.zero_grad()
            Loss = cross_entropy(pred, target)
            epochLoss = epochLoss + Loss.item()

            Loss.backward()
            optimizer.step()

        modelState = {
            "experimentName": experiment,
            "dataFile": dataFile,
            "Epoch": STEP + 1,
            "State_Dictionary": model.state_dict(),
            "Optimizer": optimizer.state_dict(),
            "batchSize": BATCHSIZE,
            "numEpoch": EPOCH,
            "workers": WORKERS,
            "learningRate": optimizer.param_groups[0]['lr'],
            "weightDecay": WEIGHTDECAY,
        }

        print("Epoch " + str(STEP + 1) + " Training Loss: " + str(epochLoss / len(trainData)) + " | Learning Rate: " + str(optimizer.param_groups[0]['lr']))
        saveCheckPoint(model, device, modelState)


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True, type=str)
parser.add_argument("--numClasses", default=100, type=int)
parser.add_argument("--dataFile", default="../data/train_cls.txt", type=str)

parser.add_argument("--root", default="../data", type=str)
parser.add_argument("--numEpoch", default=100, type=int)
parser.add_argument("--batchSize", default=512, type=int)
parser.add_argument("--workers", default=2, type=int)
parser.add_argument("--learningRate", default=1e-3, type=float)
parser.add_argument("--pretrain", action="store_true")
parser.add_argument("--weightDecay", default=0.0005, type=float)

parser.set_defaults(pretrain=False)
args = parser.parse_args()

if __name__ == "__main__":
    train(args)

