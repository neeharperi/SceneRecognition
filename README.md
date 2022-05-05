# Scene Recognition with Limited Data

Neehar Peri, Kush Singh, Uksang Yoo, Sunyu Wang

## Abstract 
In recent years, deep learning techniques trained on increasingly large datasets have brought about significant improvements in scene recognition and image classification. However, the performance of these novel techniques have not been extensively studied on small datasets, with many common deep learning models requiring millions of images to converge. Many of these large models have been pre-trained on large datasets for image recognition and classification tasks. These models are likely to have richer features than shallower light weight models due to the volume of training data, despite their different training objectives. In this paper, we present a novel approach that combines existing pre-trained feature extractors with light weight classifiers. These models are evaluated on two novel datasets: Places100, a subset of the Places365 scene classification dataset and Open-Places100, a derivative of Places100 to study a model's ability to differentiate between in-domain data and open-set examples. We first establish a baseline using ResNet-18 trained and evaluated on both datasets, measuring the accuracy of our end-to-end trained baseline. Motivated by the poor performance baseline, we propose using pretrained feature-classifier pairs to improve upon the baseline. We study VIT, CLIP, and ResNet pretrained features and pair these with neural network, SVM, and XGBoost classifiers. Lastly, since each set of pretrained feature-classifier pairs has unique failure modes, we propose a self-training framework to use the majority vote of our nine feature-classifier pairs to weakly label a larger dataset. The results show that our self-trained network improves performance compared to the pretrained feature - lightweight classifier combinations trained on small datasets, showing promise for semi-supervised applications where large sets of unlabeled data are available. 



## Data Visualization
* Under `viz` directory run `python visualizePCA.py` to get PCA visualization of 10 select classes 
* Under `viz` directory run `python visualizeTSNE.py` to get PCA visualization of 10 select classes 

## Training
1. For extracting and saving pre-trained features on which the classifiers are trained on, run `python extractFeatures.py` in the 'models' directory.  
2. To ease training the classifier models, run `./train.sh` in `train` directory after running `chmod +x train.sh`
3. (Optional) you may also train the models seperatly in the 'train' directory run `python -MODEL.py- --experiment FEATURE_MODEL_openset/standard --featureExtractor FEATURE` where MODEL, FEATURE, OPENSET/STANDARD are selected accordingly

All models are saved in `model`.

## Evaluation
1. To ease evaluating the classifier models, run `./evaluate.sh` in 'evaluate' directory
2. `confusion.py`, `plot.py` provide result visualizations
3. individual results can be gathered as before
