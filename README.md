# Scene Recognition with Limited Data

Neehar Peri, Kush Singh, Uksang Yoo, Sunyu Wang

## Abstract 
In recent years, deep learning techniques trained on increasingly large datasets have brought about significant advancements in scene recognition and image classification. However, the performance of these novel techniques have not been verified on small datasets, with many common deep learning models needing upwards of millions of images to fully train. Many of these large models have been pre-trained on large corpuses for image recognition and classification. These models are likely to have richer features than shallower light weight models due to the volume of data that they were provided, despite their different training objectives. In this paper, we present a novel approach that combines existing pre-trained feature extractors that can leverage the pretraining of these models on large amounts of data with limited compute overhead with light weight classification layers. These models are evaluated on two novel datasets we introduce: a subset of the Places365 dataset (a dataset used for classifying scenes in images) and a novel open-set dataset to study these model's abilities to differentiate between in-domain data and open-set examples. We compare these approaches with the baseline model of ResNet-18 trained and evaluated on both datasets, measuring the accuracy of our baseline and each model. Motivated by the results that indicated the proposed feature-classifier combinations' performances differed on each scene classes, we propose a self-learning framework to address challenges of generating large sets of labeled data. The results showed that our self-trained network improved performance compared to the feature-classifier combinations especially with ResNet features, showing promise for applications where large sets of pre-labeled data are not available.



##Data Visualization
1. Under `viz` directory run `python visualizePCA.py` to get PCA visualization of 10 select classes 
2. Under `viz` directory run `python visualizeTSNE.py` to get PCA visualization of 10 select classes 

## Training
1. For extracting and saving pre-trained features on which the classifiers are trained on, run `python extractFeatures.py` in the 'models' directory.  
2. To ease training the classifier models, run `./train.sh` in `train` directory after running `chmod +x train.sh`
3. (Optional) you may also train the models seperatly in the 'train' directory run `python -MODEL.py- --experiment FEATURE_MODEL_openset/standard --featureExtractor FEATURE` where MODEL, FEATURE, OPENSET/STANDARD are selected accordingly

All models are saved in `model`.

## Evaluation
1. To ease evaluating the classifier models, run `./evaluate.sh` in 'evaluate' directory
2. `confusion.py`, `plot.py` provide result visualizations
3. individual results can be gathered as before
