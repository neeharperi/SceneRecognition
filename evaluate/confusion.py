import os 
import argparse 
import pdb; 
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
predFile = open("xgb/clip_xgb_standard/xgb_eval.txt", "r")
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
selected_scenes = np.arange(0,100,10).astype(int)
keylen = len(pred.keys())

y_true = np.zeros((keylen))
y_pred = np.zeros((keylen))
i =0

for filename in pred.keys():
    y_true[i] = gt[filename]

    y_pred[i] = pred[filename]
    i+=1
listkeys =list(pred.keys())

label_cm = np.take(listkeys, selected_scenes)

cm = confusion_matrix(y_true, y_pred)
labeling = ["locker_room" ,
 "rainforest" ,
  "auditorium" ,
   "river",
    "bus_interior" ,
    "bazaar-outdoor",
    "kindergarden_classroom",
    "park",
    "swimming_pool",
    "cemetery"]
cm_short = np.take(cm, selected_scenes, axis = 0)
cm_short = np.take(cm_short, selected_scenes, axis = 1)
print(cm_short.shape)

df_cm = pd.DataFrame(cm)
plt.figure(figsize = (10,7))
sn.heatmap(df_cm)
plt.show()
