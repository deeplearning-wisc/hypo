NAME=$1
ID_DATASET=CIFAR-10 
ID_LOC=datasets/CIFAR10
OOD_LOC=datasets/small_OOD_dataset


python eval_hypo.py \
        --epoch 500 \
        --model resnet18 \
        --head mlp \
        --gpu 0 \
        --in_dataset ${ID_DATASET} \
        --id_loc ${ID_LOC} \
        --ood_loc ${OOD_LOC} \
        --name ${NAME}
