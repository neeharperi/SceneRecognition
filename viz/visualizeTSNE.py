from sklearn.manifold import TSNE
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
dirFeat = "../data/features/train/"
dirFeatTest = "../data/features/val/"
vitFeats = []
clipFeats =[]
rnFeats = []
labels = []
label_names =[]
#Loading data into lists
label = 0
#We gotta cut classes to cut nonsense
selected_scenes = np.arange(0,100,10)
for sceneName in os.listdir(dirFeat):
    if label in selected_scenes:
        for imageName in os.listdir(dirFeat+sceneName):
            loadedFeats = pickle.load(open(dirFeat+sceneName+"/"+imageName, 'rb'))
            vitFeats.append(loadedFeats['VIT'])
            clipFeats.append(loadedFeats['CLIP'])
            rnFeats.append(loadedFeats['RESNET'])
            labels.append(label)
        for imageName in os.listdir(dirFeatTest+sceneName):
            loadedFeats = pickle.load(open(dirFeatTest+sceneName+"/"+imageName, 'rb'))
            vitFeats.append(loadedFeats['VIT'])
            clipFeats.append(loadedFeats['CLIP'])
            rnFeats.append(loadedFeats['RESNET'])
            labels.append(label)
        label_names.append(sceneName)
    label += 1

vitarr = np.array(vitFeats)
rnarr = np.array(rnFeats)
clioparr = np.array(clipFeats)
labels = np.array(labels)
#PCA with 2 PCs
tsne = TSNE(n_components=2, random_state=0)
tsne_proj = tsne.fit_transform(vitFeats)
#print(pc_proj.shape)

plt.style.use('seaborn-whitegrid')
plt.figure(figsize = (10,6))
c_map = plt.cm.get_cmap('jet', 10)
plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], s = 15,
            cmap = c_map , c = labels)
cb = plt.colorbar()
loc = np.arange(0,max(labels-1),max(labels)/10)
cb.set_ticks(loc)
cb.set_ticklabels(label_names)
plt.xlabel('t-SNE Feature 1') , plt.ylabel('t-SNE Feature 2')
plt.show()