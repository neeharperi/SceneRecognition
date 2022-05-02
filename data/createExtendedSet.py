import os
import json
import pdb

def get_cls_set():
    cls_set = set()
    with open("train_cls.txt", "r") as f:
        for line in f.readlines():
            cls_set.add(line.split(" ")[1].split("\n")[0])
    with open("val_cls.txt", "r") as f:
        for line in f.readlines():
            cls_set.add(line.split(" ")[1].split("\n")[0])
    return cls_set

def main():
    num_cls = 100 
    img_per_cls = 100

    root = "places365_standard"
    splits = ["train", "val"]
    cls_map = json.load(open("cls_map.json", "r"))
    places_100_files = get_cls_set()
    for split in splits:
        file = open(split + "_extended_cls.txt", "w")
        for id, cls_name in enumerate(cls_map.values()):
            imgs = list(os.listdir(root + "/" + split + "/" + cls_name))
            num_found = 0
            for img in imgs:
                imgPath = root + "/" + split + "/" + cls_name + "/" + img
                if imgPath not in places_100_files:
                    file.write(str(id) + " " + imgPath + "\n")
                    num_found += 1
                    if num_found >= img_per_cls: 
                        break
        file.close()

    with open('cls_map.json', 'w') as f:
        json.dump(cls_map, f)

if __name__ == "__main__":
    main()