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
    def __init__(self, file, task, root, featureExtractor):
        self.file = open(file)
        self.root = root
        self.featureExtractor = featureExtractor
        self.featureFiles = []

        for line in self.file:
            ID, imgFile = line.split(" ")
            ID = int(ID)
            featureFile = imgFile.strip("\n").replace("places365_standard", "features").replace("jpg", "pkl")

            if "openset" in task:
                if ID < 50:
                    pass 

                elif ID >= 50 and ID < 75:
                    ID = 50

                else:
                    continue 

            self.featureFiles.append((ID, featureFile))

    def __len__(self):
        return len(self.featureFiles)

    def __getitem__(self, index):
        ID, featureFile = self.featureFiles[index]
        feats = pickle.load(open(self.root + "/" + featureFile, "rb"))[self.featureExtractor]

        feats = torch.tensor(feats).float()
        ID = torch.tensor(ID)

        return feats, ID

class NN(nn.Module):
    def __init__(self, numFeats, numClasses):
        super(NN, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(numFeats, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, numClasses))
    
    def forward(self, feats):
        return self.classifier(feats)


def saveCheckPoint(model, device, modelState):
    fileName = "../models/nn/" + modelState["experimentName"] + "/modelCheckPoint" + str(modelState["Epoch"]) + ".pth.tar"
    torch.save(modelState, fileName)

    return model.to(device)

def train(args):
    experiment = args.experiment
    featureExtractor = args.featureExtractor
    numClasses = args.numClasses
    dataFile = args.dataFile
    root = args.root
    BATCHSIZE = args.batchSize
    EPOCH = args.numEpoch
    WORKERS = args.workers
    LR = args.learningRate
    WEIGHTDECAY = args.weightDecay

    if featureExtractor == "CLIP":
        numFeatures = 512
    elif featureExtractor == "VIT":
        numFeatures = 768
    elif featureExtractor == "RESNET":
        numFeatures = 2048
    else:
        assert False, "Invalid Feature Extractor"

    if not os.path.isdir("../models/nn/" + experiment + "/"):
        os.makedirs("../models/nn/" + experiment + "/")

    print("Using Parameters:")
    pprint(vars(args))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TrainDataset = SceneRecognitionDataLoader(dataFile, experiment, root, featureExtractor)
    trainData = DataLoader(TrainDataset, batch_size=BATCHSIZE, shuffle=True, num_workers=WORKERS)

    model = NN(numFeatures, numClasses)
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
            feats, target = data

            feats = feats.to(device)
            target = target.to(device)

            pred = model(feats)

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
parser.add_argument("--featureExtractor", required=True, type=str)

parser.add_argument("--numClasses", default=100, type=int)
parser.add_argument("--dataFile", default="../data/train_cls.txt", type=str)

parser.add_argument("--root", default="../data", type=str)
parser.add_argument("--numEpoch", default=100, type=int)
parser.add_argument("--batchSize", default=512, type=int)
parser.add_argument("--workers", default=2, type=int)
parser.add_argument("--learningRate", default=1e-3, type=float)
parser.add_argument("--weightDecay", default=0.0005, type=float)

args = parser.parse_args()

if __name__ == "__main__":
    train(args)
