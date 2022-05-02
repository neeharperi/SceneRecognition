import matplotlib.pyplot as plt
import numpy as np 

def evaluate(file_name, eval_type):
    predFile = open(file_name, "r")
    gtFile = open("../data/val_cls.txt", "r")
    pred, gt = {}, {}

    for line in predFile:
        ID, fileName = line.split(" ")
        fileName = fileName.strip("\n")
        ID = float(ID)

        pred[fileName] = int(ID)

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
        return acc
    
    else:
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
        return open_acc, closed_acc


def get_accuracy(file_name, mode):
    if mode == "standard":
        return evaluate(file_name, "standard")
    elif mode == "openset":
        return evaluate(file_name, "openset")[0]
    else:
        return evaluate(file_name, "openset")[1]

def plot_models():
    EVAL_MODES = ["standard", "openset", "closed_set"]
    SINGLE_MODELS = ["kmeans", "svm", "xgb"]
    EXTRACTOR_LAYERS = ["clip", "resnet", "vit"]
    NUM_EPOCHS = 100
    
    for mode in EVAL_MODES:
        file_mode = mode if mode!="closed_set" else "openset"
        nn_resnet_accs = [get_accuracy(f"baseline/resnet_{file_mode}/modelCheckPoint{str(i+1)}.txt", mode) for i in range(NUM_EPOCHS)]
        nn_pretrained_resnet_accs = [get_accuracy(f"baseline/resnet_{file_mode}_pretrain/modelCheckPoint{str(i+1)}.txt", mode) for i in range(NUM_EPOCHS)]

        for layer in EXTRACTOR_LAYERS:
            nn_epoch_accs = [get_accuracy(f"nn/{str(layer)}_nn_{file_mode}/modelCheckPoint{str(i+1)}.txt", mode) for i in range(NUM_EPOCHS)]
            model_mapping = {}
            for model in SINGLE_MODELS:
                file_name = f"{model}/{layer}_{model}_{file_mode}/{model}_eval.txt"
                model_mapping[model] = get_accuracy(file_name, mode)
            
            plt.clf()
            plt.plot([i+1 for i in range(NUM_EPOCHS)], nn_resnet_accs, label="resnet_baseline")
            plt.plot([i+1 for i in range(NUM_EPOCHS)], nn_pretrained_resnet_accs, label="resnet_pretrained")
            plt.plot([i+1 for i in range(NUM_EPOCHS)], nn_epoch_accs, label="nn")
            COLORS = ["green", "red", "orange"]
            c = 0
            for model in model_mapping:
                plt.axhline(y=model_mapping[model], linestyle='-', label=model, color=COLORS[c])
                c += 1
            plt.title(f"Comparison for {layer} extraction layer in {mode} mode")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.xlim((0, 100))
            plt.savefig(f"../plots/{layer}_{mode}.eps")
            plt.savefig(f"../plots/{layer}_{mode}.png")


if __name__ == "__main__":
    plot_models()