import os
import json
import pdb

num_cls = 100 
img_per_cls = 10

root = "places365_standard"
splits = ["train", "val"]
cls_list = list(os.listdir(root + "/train/"))[:num_cls]
cls_map = {}
for split in splits:
    file = open(split + "_cls.txt", "w")
    for id, cls_name in enumerate(cls_list):
        cls_map[id] = cls_name
        imgs = list(os.listdir(root + "/" + split + "/" + cls_name))[:img_per_cls]
        for img in imgs:
            imgPath = root + "/" + split + "/" + cls_name + "/" + img
            file.write(str(id) + " " + imgPath + "\n")
    file.close()

with open('cls_map.json', 'w') as f:
    json.dump(cls_map, f)