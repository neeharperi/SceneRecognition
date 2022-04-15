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


python classification.py --predFile kmeans/vit_kmeans_standard/kmeans_eval.txt --eval_type standard
python classification.py --predFile kmeans/vit_kmeans_openset/kmeans_eval.txt --eval_type openset

python classification.py --predFile kmeans/clip_kmeans_standard/kmeans_eval.txt --eval_type standard
python classification.py --predFile kmeans/clip_kmeans_openset/kmeans_eval.txt --eval_type openset

python classification.py --predFile kmeans/resnet_kmeans_standard/kmeans_eval.txt --eval_type standard
python classification.py --predFile kmeans/resnet_kmeans_openset/kmeans_eval.txt --eval_type openset

python classification.py --predFile svm/vit_svm_standard/svm_eval.txt --eval_type standard
python classification.py --predFile svm/vit_svm_openset/svm_eval.txt --eval_type openset

python classification.py --predFile svm/clip_svm_standard/svm_eval.txt --eval_type standard
python classification.py --predFile svm/clip_svm_openset/svm_eval.txt --eval_type openset

python classification.py --predFile svm/resnet_svm_standard/svm_eval.txt --eval_type standard
python classification.py --predFile svm/resnet_svm_openset/svm_eval.txt --eval_type openset

python classification.py --predFile xgb/vit_xgb_standard/xgb_eval.txt --eval_type standard
python classification.py --predFile xgb/vit_xgb_openset/xgb_eval.txt --eval_type openset

python classification.py --predFile xgb/clip_xgb_standard/xgb_eval.txt --eval_type standard
python classification.py --predFile xgb/clip_xgb_openset/xgb_eval.txt --eval_type openset

python classification.py --predFile xgb/resnet_xgb_standard/xgb_eval.txt --eval_type standard
python classification.py --predFile xgb/resnet_xgb_openset/xgb_eval.txt --eval_type openset
