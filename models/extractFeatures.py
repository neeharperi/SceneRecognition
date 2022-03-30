import os
import torch 
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import clip 
from pytorch_pretrained_vit import ViT
import pickle
from tqdm import tqdm
from PIL import Image

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.feats = nn.Sequential(*list(models.resnet101(pretrained=True).children())[:-1])

    def forward(self, img):
        return self.feats(img)


device = "cuda" if torch.cuda.is_available() else "cpu"

vit_model =  ViT('B_32_imagenet1k', pretrained=True)
del vit_model.fc
clip_model, preprocess = clip.load("ViT-B/32", device=device)
resnet_model = ResNet().to(device)

vit_model.eval()
clip_model.eval()
resnet_model.eval()

splits = ["train", "val"]

save_root = "../data/features"
if not os.path.isdir(save_root):
    os.makedirs(save_root)
    
for split in splits:
    data_root = "../data/"
    fileNames = open(data_root + split + "_cls.txt")

    for line in tqdm(fileNames):
        _, loc = line.split()
        imgFile = data_root + loc
        vit_img = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(), transforms.Normalize(0.5, 0.5),])(Image.open(imgFile).convert("RGB")).unsqueeze(0)
        clip_img = preprocess(Image.open(imgFile).convert("RGB")).unsqueeze(0).to(device)
        resnet_img = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(Image.open(imgFile).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            vit_features = vit_model(vit_img)[0, 0].cpu().numpy()
            clip_features = clip_model.encode_image(clip_img)[0].cpu().numpy()
            resnet_features = resnet_model(resnet_img)[0,:,0,0].cpu().numpy()

            featureFile = imgFile.replace("places365_standard", "features").replace(".jpg", ".pkl")
            folder = "/".join(featureFile.split("/")[:-1])

            if not os.path.isdir(folder):
                os.makedirs(folder)

            feats = {"VIT" : vit_features,
                     "CLIP" : clip_features,
                     "RESNET" : resnet_features}

            pickle.dump(feats, open(featureFile, "wb"))

            


