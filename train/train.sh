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
