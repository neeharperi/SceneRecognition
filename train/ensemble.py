import os 
import argparse
import os
from pprint import pprint
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import pickle
from scipy import stats
import torch
from torch.utils.data import Dataset as TorchDataset
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter

class SceneRecognitionDataLoaderTorch(TorchDataset):
    def __init__(self, file, root, featureExtractor):
        self.file = open(file)
        self.root = root
        self.featureExtractor = featureExtractor
        self.featureFiles = []
        self.imageFiles = []

        for line in self.file:
            ID, imgFile = line.split(" ")
            ID = int(ID)

            imgFile = imgFile.strip("\n")
            self.imageFiles.append((ID, imgFile))
        
            featureFile = imgFile.replace("places365_standard", "features").replace("jpg", "pkl")
            self.featureFiles.append((ID, featureFile))

    def __len__(self):
        return len(self.featureFiles)

    def __getitem__(self, index):
        ID, featureFile = self.featureFiles[index]
        _, imgFile = self.imageFiles[index]
        feats = pickle.load(open(self.root + "/" + featureFile, "rb"))[self.featureExtractor]

        feats = torch.tensor(feats).float()
        ID = torch.tensor(ID)

        return feats, ID, imgFile

class NN(nn.Module):
    def __init__(self, numFeats, numClasses):
        super(NN, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(numFeats, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, numClasses))
    
    def forward(self, feats):
        return self.classifier(feats)


class SceneRecognitionDataLoaderSKLearn():
    def __init__(self, file, root, featureExtractor):
        self.file = open(file)
        self.root = root
        self.featureExtractor = featureExtractor
        self.featureFiles = []
        self.imageFiles = []

        for line in self.file:
            ID, imgFile = line.split(" ")
            ID = int(ID)

            imgFile = imgFile.strip("\n")
            self.imageFiles.append((ID, imgFile))
        
            featureFile = imgFile.replace("places365_standard", "features").replace("jpg", "pkl")
            self.featureFiles.append((ID, featureFile))

    def get_x_y_pts(self):
        feats_pts = []
        id_pts = []
        img_files = []
        for index in range(len(self.featureFiles)):
            ID, featureFile = self.featureFiles[index]
            _, imgFile = self.imageFiles[index]
            feats = pickle.load(open(self.root + "/" + featureFile, "rb"))[self.featureExtractor]

            feats = np.array(feats)#.astype(np.float)
            #print(feats.shape)
            ID = np.array(ID)

            feats_pts.append(feats)
            id_pts.append(ID)
            img_files.append(imgFile)

        return feats_pts, id_pts, img_files

def remove_module(state_dict):
    new_dict = dict()
    for key in state_dict:
        new_key = key.split("module.")[1] if "module." in key else key
        new_dict[new_key] = state_dict[key]
    return new_dict

def get_models():
    sklearn_models = dict()
    pytorch_models = dict()

    sklearn_models["svm_clip"] = pickle.load(open('../models/svm/clip_svm_standard/svm.pkl', 'rb'))
    sklearn_models["svm_resnet"] = pickle.load(open('../models/svm/resnet_svm_standard/svm.pkl', 'rb'))
    sklearn_models["svm_vit"] = pickle.load(open('../models/svm/vit_svm_standard/svm.pkl', 'rb'))
    sklearn_models["kmeans_clip"] = pickle.load(open('../models/kmeans/clip_kmeans_standard/kmeans.pkl', 'rb'))
    sklearn_models["kmeans_resnet"] = pickle.load(open('../models/kmeans/resnet_kmeans_standard/kmeans.pkl', 'rb'))
    sklearn_models["kmeans_vit"] = pickle.load(open('../models/kmeans/vit_kmeans_standard/kmeans.pkl', 'rb'))
    sklearn_models["xgb_clip"] = pickle.load(open('../models/xgb/clip_xgb_standard/xgb.pkl', 'rb'))
    sklearn_models["xgb_resnet"] = pickle.load(open('../models/xgb/resnet_xgb_standard/xgb.pkl', 'rb'))
    sklearn_models["xgb_vit"] = pickle.load(open('../models/xgb/vit_xgb_standard/xgb.pkl', 'rb'))

    nn_clip = NN(512, 100)
    clip_state = torch.load('../models/nn/clip_nn_standard/modelCheckPoint100.pth.tar', map_location ='cpu')["State_Dictionary"]
    nn_clip.load_state_dict(remove_module(clip_state))

    nn_resnet = NN(2048, 100)
    resnet_state = torch.load('../models/nn/resnet_nn_standard/modelCheckPoint100.pth.tar', map_location ='cpu')["State_Dictionary"]
    nn_resnet.load_state_dict(remove_module(resnet_state))

    nn_vit = NN(768, 100)
    vit_state = torch.load('../models/nn/vit_nn_standard/modelCheckPoint99.pth.tar', map_location ='cpu')["State_Dictionary"]
    nn_vit.load_state_dict(remove_module(vit_state))

    pytorch_models["nn_clip"] = nn_clip
    pytorch_models["nn_resnet"] = nn_resnet
    pytorch_models["nn_vit"] = nn_vit

    return sklearn_models, pytorch_models

def get_sklearn_counts(sklearn_models, imgFile, root):
    counts = dict()

    for model in sklearn_models:
        modelName = model.split("_")[0]
        featureExtractor = model.split("_")[1].upper()
        Dataset = SceneRecognitionDataLoaderSKLearn(imgFile, root, featureExtractor)
        feats, _, imgs = Dataset.get_x_y_pts()
        if modelName == "xgb":
            feats = np.array(feats)
        preds = sklearn_models[model].predict(feats)

        for predID, predFile in zip(preds, imgs):
            if predFile not in counts:
                counts[predFile] = list()
            counts[predFile].append(predID)
        
    return counts

def get_pytorch_counts(pytorch_models, currDatasetFile, root):
    counts = dict()
    vit_nn_cls = dict()

    for model in pytorch_models:
        pytorch_models[model].eval()
        featureExtractor = model.split("_")[1].upper()
        Dataset = SceneRecognitionDataLoaderTorch(currDatasetFile, root, featureExtractor)
        valData = DataLoader(Dataset, batch_size=512, shuffle=False, num_workers=2)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():   
            for data in valData:
                feats, target, imgFile = data

                feats = feats.to(device)
                target = target.to(device)

                pred = torch.argmax(pytorch_models[model](feats), axis=1)
                for predID, predFile in zip(pred, imgFile):
                    if predFile not in counts:
                        counts[predFile] = list()
                    if model == "nn_vit":
                        vit_nn_cls[predFile] = predID.item()
                    counts[predFile].append(predID.item())
    
    return counts, vit_nn_cls

def evaluate(args):
    imgFile = args.imgFile
    root = args.root

    if not os.path.isdir("../data/ensemble/"):
        os.makedirs("../data/ensemble/")

    fileName = "../data/ensemble/ensemble_preds.txt"

    file = open(fileName, "w")    
    sklearn_models, pytorch_models = get_models()

    counts_sklearn = get_sklearn_counts(sklearn_models, imgFile, root)
    counts_pytorch, vit_nn_cls = get_pytorch_counts(pytorch_models, imgFile, root)
    aggregated_counts = dict()
    for filename in counts_sklearn:
        aggregated_counts[filename] = counts_sklearn[filename] + counts_pytorch[filename]
    
    for filename in aggregated_counts:
        counter = Counter(aggregated_counts[filename])
        max_count = max(counter.values())
        mode = [k for k,v in counter.items() if v == max_count]

        pred_cls = mode[0] if len(mode) == 1 else vit_nn_cls[filename]
        file.write(str(pred_cls) + " " + filename + "\n")                

    file.close()

parser = argparse.ArgumentParser()
parser.add_argument("--imgFile", default="../data/train_extended_cls.txt", type=str)
parser.add_argument("--root", default="../data", type=str)

args = parser.parse_args()

if __name__ == "__main__":
    evaluate(args)
