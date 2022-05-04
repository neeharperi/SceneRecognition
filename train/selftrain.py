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
    def __init__(self, file, task, root, mode, transform=None):
        self.file = open(file)
        self.transform = transform
        self.mode = mode
        self.root = root
        self.imageFiles = []
        self.featureFiles = []

        for line in self.file:
            ID, imgFile = line.split(" ")
            ID = int(ID)

            if mode == "train" and "openset" in task:
                if ID < 50:
                    pass 

                elif ID >= 50 and ID < 75:
                    ID = 50

                else:
                    continue 

            if mode == "val" and "openset" in task:
                if ID < 50:
                    pass 

                elif ID >= 50:
                    ID = 50

            imgFile = imgFile.strip("\n")
            featureFile = imgFile.replace("places365_standard", "features").replace("jpg", "pkl")

            self.imageFiles.append((ID, imgFile))
            self.featureFiles.append((ID, featureFile))

    def __len__(self):
        return len(self.imageFiles)

    def __getitem__(self, index):
        ID, imgFile = self.imageFiles[index]
        _, featureFile = self.featureFiles[index]
        img = Image.open(self.root + "/" + imgFile).convert("RGB")
        feats = pickle.load(open(self.root + "/" + featureFile, "rb"))["CLIP"]

        feats = torch.tensor(feats).float()

        if self.transform:
            img = self.transform(img)

        ID = torch.tensor(ID)

        if self.mode == "train":
            return img, feats, ID
        else:
            return img, feats, ID, imgFile

class ResNet(nn.Module):
    def __init__(self, numClasses, pretrain):
        super(ResNet, self).__init__()
        self.feats = nn.Sequential(*list(models.resnet18(pretrained=pretrain).children())[:-1])
        self.classifier = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, numClasses))
    
    def forward(self, img, feats):
        preds = self.classifier(self.feats(img).reshape(-1, 512) + feats)

        return preds

def saveCheckPoint(model, device, modelState):
    fileName = "../models/ensemble/" + modelState["experimentName"] + "/modelCheckPoint" + str(modelState["Epoch"]) + ".pth.tar"
    torch.save(modelState, fileName)


def evaluate(args, model, device, modelState):
    experiment = args.experiment
    valFile = args.valFile
    root = args.root
    BATCHSIZE = args.batchSize
    WORKERS = args.workers

    if not os.path.isdir("../evaluate/ensemble/" + experiment + "/"):
        os.makedirs("../evaluate/ensemble/" + experiment + "/")

    fileName = "../evaluate/ensemble/" + modelState["experimentName"] + "/modelCheckPoint" + str(modelState["Epoch"]) + ".txt"

    file = open(fileName, "w")

    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ValDataset = SceneRecognitionDataLoader(valFile, experiment, root, "val", transform)
    valData = DataLoader(ValDataset, batch_size=BATCHSIZE, shuffle=False, num_workers=WORKERS)

    model.eval()

    total, correct = 0, 0
    with torch.no_grad():
        for data in valData:
            img, feats, target, imgFile = data

            img = img.to(device)
            feats = feats.to(device)
            target = target.to(device)

            pred = torch.argmax(model(img, feats), axis=1)
            
            correct += torch.sum(pred == target)
            total += target.shape[0]

            for predID, predFile in zip(pred, imgFile):
                file.write(str(predID.item()) + " " + predFile + "\n")                

    file.close()

    acc = correct / total 

    print("Validation Accuracy: {}".format(100 * acc))

def train(args):
    experiment = args.experiment
    numClasses = args.numClasses
    trainFile = args.trainFile
    root = args.root
    BATCHSIZE = args.batchSize
    EPOCH = args.numEpoch
    WORKERS = args.workers
    LR = args.learningRate
    WEIGHTDECAY = args.weightDecay
    pretrain = args.pretrain

    if not os.path.isdir("../models/ensemble/" + experiment + "/"):
        os.makedirs("../models/ensemble/" + experiment + "/")

    print("Using Parameters:")
    pprint(vars(args))

    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TrainDataset = SceneRecognitionDataLoader(trainFile, experiment, root, "train", transform)
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
            img, feats, target = data

            img = img.to(device)
            feats = feats.to(device)
            target = target.to(device)
            pred = model(img, feats)

            optimizer.zero_grad()
            Loss = cross_entropy(pred, target)

            epochLoss = epochLoss + Loss.item()

            Loss.backward()
            optimizer.step()

        modelState = {
            "experimentName": experiment,
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
        evaluate(args, model, device, modelState)
        saveCheckPoint(model, device, modelState)


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True, type=str)
parser.add_argument("--numClasses", default=100, type=int)
parser.add_argument("--trainFile", default="../data/ensemble/ensemble_preds.txt", type=str)
parser.add_argument("--valFile", default="../data/val_cls.txt", type=str)

parser.add_argument("--root", default="../data", type=str)
parser.add_argument("--numEpoch", default=100, type=int)
parser.add_argument("--batchSize", default=128, type=int)
parser.add_argument("--workers", default=4, type=int)
parser.add_argument("--learningRate", default=1e-3, type=float)
parser.add_argument("--weightDecay", default=0.0005, type=float)
parser.add_argument("--pretrain", action="store_true")

parser.set_defaults(pretrain=False)
args = parser.parse_args()

if __name__ == "__main__":
    train(args)

