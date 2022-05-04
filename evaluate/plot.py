import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

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
        acc = 0.5 * closed_acc + 0.5 * open_acc
        return acc


def get_accuracy(file_name, mode):
    if mode == "standard":
        return evaluate(file_name, "standard")
    else:
        return evaluate(file_name, "openset")

def list_to_increasing_list(arr, epochs):
    x_arr = []
    y_arr = []
    max_val = arr[0]
    for i in range(epochs):
        if arr[i]>= max_val:
            x_arr.append(i+1)
            y_arr.append(arr[i])
            max_val = arr[i]
        elif i == epochs-1:
            x_arr.append(epochs)
            y_arr.append(max_val)
    return x_arr, y_arr

def plot_models():
    EVAL_MODES = ["standard", "openset"]
    SINGLE_MODELS = ["kmeans", "svm", "xgb"]
    EXTRACTOR_LAYERS = ["clip", "resnet", "vit"]
    NUM_EPOCHS = 100
    csv_arr = []

    for mode in EVAL_MODES:
        nn_resnet_accs = [get_accuracy(f"baseline/resnet_{mode}/modelCheckPoint{str(i+1)}.txt", mode) for i in range(NUM_EPOCHS)]
        nn_pretrained_resnet_accs = [get_accuracy(f"baseline/resnet_{mode}_pretrain/modelCheckPoint{str(i+1)}.txt", mode) for i in range(NUM_EPOCHS)]
        nn_resnet_self_train_accs = [get_accuracy(f"ensemble/resnet_selftrain_{mode}/modelCheckPoint{str(i+1)}.txt", mode) for i in range(NUM_EPOCHS)]
        nn_pretrained_resnet_self_train_accs = [get_accuracy(f"ensemble/resnet_selftrain_{mode}_pretrain/modelCheckPoint{str(i+1)}.txt", mode) for i in range(NUM_EPOCHS)]
        
        resnet_baseline_x, resnet_baseline_y = list_to_increasing_list(nn_resnet_accs, NUM_EPOCHS)
        resnet_pretrained_baseline_x, resnet_pretrained_baseline_y = list_to_increasing_list(nn_pretrained_resnet_accs, NUM_EPOCHS)
        resnet_selftrain_x, resnet_selftrain_y = list_to_increasing_list(nn_resnet_self_train_accs, NUM_EPOCHS)
        resnet_selftrain_pretrained_x, resnet_selftrain_pretrained_y = list_to_increasing_list(nn_pretrained_resnet_self_train_accs, NUM_EPOCHS)

        csv_arr.append({"mode": mode, "layer": "N/A", "model": "resnet_baseline", "acc": resnet_baseline_y[-1]})
        csv_arr.append({"mode": mode, "layer": "N/A", "model": "resnet_baseline_pretrained", "acc": resnet_pretrained_baseline_y[-1]})
        csv_arr.append({"mode": mode, "layer": "N/A", "model": "resnet_selftrain", "acc": resnet_selftrain_y[-1]})
        csv_arr.append({"mode": mode, "layer": "N/A", "model": "resnet_selftrain_pretrained", "acc": resnet_selftrain_pretrained_y[-1]})

        for layer in EXTRACTOR_LAYERS:
            nn_epoch_accs = [get_accuracy(f"nn/{str(layer)}_nn_{mode}/modelCheckPoint{str(i+1)}.txt", mode) for i in range(NUM_EPOCHS)]
            model_mapping = {}
            for model in SINGLE_MODELS:
                file_name = f"{model}/{layer}_{model}_{mode}/{model}_eval.txt"
                model_mapping[model] = get_accuracy(file_name, mode)
            
            plt.clf()
            plt.plot(resnet_baseline_x, resnet_baseline_y, label="resnet_baseline")
            plt.plot(resnet_pretrained_baseline_x, resnet_pretrained_baseline_y, label="resnet_pretrained")
            plt.plot(resnet_selftrain_x, resnet_selftrain_y, label="resnet_selftrain")
            plt.plot(resnet_selftrain_pretrained_x, resnet_selftrain_pretrained_y, label="resnet_selftrain_pretrained")
            nn_x, nn_y = list_to_increasing_list(nn_epoch_accs, NUM_EPOCHS)
            plt.plot(nn_x, nn_y, label="nn")
            
            csv_arr.append({"mode": mode, "layer": layer, "model": "nn", "acc": nn_y[-1]})

            COLORS = ["green", "red", "orange"]
            c = 0
            for model in model_mapping:
                plt.axhline(y=model_mapping[model], linestyle='-', label=model, color=COLORS[c])
                csv_arr.append({"mode": mode, "layer": layer, "model": model, "acc": model_mapping[model]})
                c += 1
            plt.title(f"Comparison for {layer} extraction layer in {mode} mode")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.xlim((0, 100))
            plt.savefig(f"../plots/{layer}_{mode}.eps")
            plt.savefig(f"../plots/{layer}_{mode}.png")
    df = pd.DataFrame.from_dict(csv_arr) 
    df.to_csv (r'models.csv', index = False, header=True)



if __name__ == "__main__":
    plot_models()