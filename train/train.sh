python baseline.py --experiment resnet_standard
python baseline.py --experiment resnet_openset
python baseline.py --experiment resnet_standard_pretrain --pretrain
python baseline.py --experiment resnet_openset_pretrain --pretrain

python nn.py --experiment vit_nn_standard --featureExtractor VIT
python nn.py --experiment vit_nn_openset --featureExtractor VIT
python nn.py --experiment clip_nn_standard --featureExtractor CLIP
python nn.py --experiment clip_nn_openset --featureExtractor CLIP
python nn.py --experiment resnet_nn_standard --featureExtractor RESNET
python nn.py --experiment resnet_nn_openset --featureExtractor RESNET

python xgb.py --experiment vit_xgb_standard --featureExtractor VIT
python xgb.py --experiment vit_xgb_openset --featureExtractor VIT
python xgb.py --experiment clip_xgb_standard --featureExtractor CLIP
python xgb.py --experiment clip_xgb_openset --featureExtractor CLIP
python xgb.py --experiment resnet_xgb_standard --featureExtractor RESNET
python xgb.py --experiment resnet_xgb_openset --featureExtractor RESNET

python kmeans.py --experiment vit_kmeans_standard --featureExtractor VIT
python kmeans.py --experiment vit_kmeans_openset --featureExtractor VIT
python kmeans.py --experiment clip_kmeans_standard --featureExtractor CLIP
python kmeans.py --experiment clip_kmeans_openset --featureExtractor CLIP
python kmeans.py --experiment resnet_kmeans_standard --featureExtractor RESNET
python kmeans.py --experiment resnet_kmeans_openset --featureExtractor RESNET