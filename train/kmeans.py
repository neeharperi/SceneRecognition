import os 
import argparse
import os
from pprint import pprint
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import pickle
import statistics
class SceneRecognitionDataLoader():
    def __init__(self, file, task, root, mode, featureExtractor):
        self.file = open(file)
        self.root = root
        self.mode = mode
        self.featureExtractor = featureExtractor
        self.featureFiles = []
        self.imageFiles = []

        for line in self.file:
            ID, imgFile = line.split(" ")
            ID = int(ID)

            if mode == "train" and task == "openset":
                if ID < 50:
                    pass 

                elif ID >= 50 and ID < 75:
                    ID = 50

                else:
                    continue 

            if mode == "val" and task == "openset":
                if ID < 50:
                    pass 

                elif ID >= 50:
                    ID = 50

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

        if self.mode == "train":
            return feats_pts, id_pts
        else:
            return feats_pts, id_pts, img_files

def evaluate(args, model, featureExtractor ):
    experiment = args.experiment
    valFile = args.valFile
    root = args.root
    task = args.task

    if not os.path.isdir("../evaluate/kmeans/" + experiment + "/"):
        os.makedirs("../evaluate/kmeans/" + experiment + "/")

    fileName = "../evaluate/kmeans/" + experiment + "/kmeans_eval.txt"
    mapp = np.load("../models/kmeans/" + experiment + "/map.npy")
    file = open(fileName, "w")

    ValDataset = SceneRecognitionDataLoader(valFile, task, root, "val", featureExtractor)
    feats, _, imgs = ValDataset.get_x_y_pts()
    preds = model.predict(feats)
    for predID, predFile in zip(preds, imgs):
        prediction = int(float(mapp[int(predID)-1]))
        file.write(str(prediction) + " " + str(predFile) + "\n")                

    file.close()

def train(args):
    experiment = args.experiment
    featureExtractor = args.featureExtractor
    trainFile = args.trainFile
    root = args.root
    task = args.task

    if not os.path.isdir("../models/kmeans/" + experiment + "/"):
        os.makedirs("../models/kmeans/" + experiment + "/")

    print("Using Parameters:")
    pprint(vars(args))

    TrainDataset = SceneRecognitionDataLoader(trainFile, task, root, "train", featureExtractor)
    feats, labels = TrainDataset.get_x_y_pts()

    model = KMeans(n_clusters=(np.unique(labels)).shape[0])

    model.fit(feats, labels)

    feat_pred = model.predict(feats)
    print(feat_pred.size) 
    #mapp is map index: cluster number-1 element: label in true label
    mapp = np.zeros(int(feat_pred.size/10))
    for i in range(int(feat_pred.size/10)):
        prediction= max([p[0] for p in statistics._counts(feat_pred[i*10:i*10+10])])
        #prediction = mode(feat_pred[i*10:i*10+10])
        mapp[prediction-1] = i
    np.save("../models/kmeans/" + experiment + "/map.npy", mapp)
    pickle.dump(model, open("../models/kmeans/" + experiment + "/kmeans.pkl", 'wb'))

    evaluate(args, model, featureExtractor)


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True, type=str)
parser.add_argument("--featureExtractor", required=True, type=str)
parser.add_argument("--trainFile", default="../data/train_cls.txt", type=str)
parser.add_argument("--valFile", default="../data/val_cls.txt", type=str)
parser.add_argument("--task", default="default", type=str)

parser.add_argument("--root", default="../data", type=str)

args = parser.parse_args()

if __name__ == "__main__":
    train(args)
