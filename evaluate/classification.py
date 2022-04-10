import os 
import argparse 
import pdb; 

def evaluate(args):
    predFile = open(args.predFile, "r")
    gtFile = open(args.gtFile, "r")
    eval_type = args.eval_type
    pred, gt = {}, {}

    for line in predFile:
        ID, fileName = line.split(" ")
        fileName = fileName.strip("\n")
        ID = int(ID)

        pred[fileName] = ID

    for line in gtFile:
        ID, fileName = line.split(" ")
        fileName = fileName.strip("\n")
        ID = int(ID)

        if eval_type == "openset":
            if ID < 50:
                pass 

            elif ID >= 50:
                ID = 50

        gt[fileName] = ID

        
    correct, total = 0, 0
    for fileName in pred.keys():
        if pred[fileName] == gt[fileName]:
            correct = correct + 1
        
        total = total + 1

    acc = correct / total

    print("Classification Accuracy: {} %".format(100 * acc))

parser = argparse.ArgumentParser()
parser.add_argument("--predFile", type=str, required=True)
parser.add_argument("--gtFile", default="../data/val_cls.txt", type=str)
parser.add_argument("--eval_type", default="standard", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    evaluate(args)
