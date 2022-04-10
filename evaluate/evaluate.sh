python classification.py --predFile baseline/resnet_standard/modelCheckPoint50.txt --eval_type standard
python classification.py --predFile baseline/resnet_standard_pretrain/modelCheckPoint50.txt --eval_type standard
python classification.py --predFile baseline/resnet_openset/modelCheckPoint50.txt --eval_type openset
python classification.py --predFile baseline/resnet_openset_pretrain/modelCheckPoint50.txt --eval_type openset

python classification.py --predFile nn/vit_nn_standard/modelCheckPoint50.txt --eval_type standard
python classification.py --predFile nn/vit_nn_openset/modelCheckPoint50.txt --eval_type openset

python classification.py --predFile nn/clip_nn_standard/modelCheckPoint50.txt --eval_type standard
python classification.py --predFile nn/clip_nn_openset/modelCheckPoint50.txt --eval_type openset

python classification.py --predFile nn/resnet_nn_standard/modelCheckPoint50.txt --eval_type standard
python classification.py --predFile nn/resnet_nn_openset/modelCheckPoint50.txt --eval_type openset
