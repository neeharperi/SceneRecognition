import matplotlib.pyplot as plt
import numpy as np 

def evaluate(file_name):
    predFile = open(file_name, "r")
    gtFile = open("../data/val_cls.txt", "r")
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
        gt[fileName] = ID

        
    correct, total = 0, 0
    closed_correct, closed_total, open_correct, open_total = 0, 0, 0, 0

    for fileName in pred.keys():
        if pred[fileName] == gt[fileName]:
            correct = correct + 1
        
        total = total + 1

    acc = correct / total
    return acc

def plot_models():
    
    SINGLE_MODELS = ["kmeans", "xgb", "svm"]
    EXTRACTOR_LAYERS = ["clip", "resnet", "vit"]
    NUM_EPOCHS = 100
    
    nn_resnet_accs = [evaluate(f"baseline/resnet_standard/modelCheckPoint{str(i+1)}.txt") for i in range(NUM_EPOCHS)]
    # nn_pretrained_resnet_accs = [evaluate(f"baseline/resnet_standard_pretrain/modelCheckPoint{str(i+1)}.txt") for i in range(NUM_EPOCHS)]

    for layer in EXTRACTOR_LAYERS:
        # nn_epoch_accs = [evaluate(f"nn/{str(layer)}_nn_standard/modelCheckPoint{str(i+1)}.txt") for i in range(NUM_EPOCHS)]
        model_mapping = {}
        for model in SINGLE_MODELS:
            file_name = f"{model}/{layer}_{model}_standard/{model}_eval.txt"
            model_mapping[model] = evaluate(file_name)
        
        plt.clf()
        plt.plot([i+1 for i in range(NUM_EPOCHS)], nn_resnet_accs, label="resnet_baseline")
        # plt.plot([i+1 for i in range(NUM_EPOCHS+1)], nn_pretrained_resnet_accs, label="resnet_pretrained")
        # plt.plot([i+1 for i in range(NUM_EPOCHS+1)], nn_epoch_accs, label="nn")
        COLORS = ["green", "red", "orange"]
        c = 0
        for model in model_mapping:
            plt.axhline(y=model_mapping[model], linestyle='-', label=model, color=COLORS[c])
            c += 1
        plt.title(f"Comparison for {layer} extraction layer")
        plt.legend()
        plt.xlim((0, 100))
        plt.savefig(f"../plots/{layer}.png")


if __name__ == "__main__":
    plot_models()