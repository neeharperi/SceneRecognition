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
    closed_correct, closed_total, open_correct, open_total = 0, 0, 0, 0

    if eval_type == "standard":
        for fileName in pred.keys():
            if pred[fileName] == gt[fileName]:
                correct = correct + 1
            
            total = total + 1

        acc = correct / total

        print("{} | Classification Accuracy: {} %".format(args.predFile, 100 * acc))

    elif eval_type == "openset":
        for fileName in pred.keys():
            if gt[fileName] == 50:
                if pred[fileName] == gt[fileName]:
                    open_correct = open_correct + 1
                
                open_total = open_total + 1

            else:
                if pred[fileName] == gt[fileName]:
                    closed_correct = closed_correct + 1
                
                closed_total = closed_total + 1

        closed_acc = closed_correct / closed_total
        open_acc = open_correct / open_total

        print("{} | Closed-Set Accuracy: {} % | Open-Set Accuracy: {} %".format(args.predFile, 100 * closed_acc, 100 * open_acc))

    else:
        assert False, "Invalid eval_type"

parser = argparse.ArgumentParser()
parser.add_argument("--predFile", type=str, required=True)
parser.add_argument("--gtFile", default="../data/val_cls.txt", type=str)
parser.add_argument("--eval_type", default="standard", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    evaluate(args)
