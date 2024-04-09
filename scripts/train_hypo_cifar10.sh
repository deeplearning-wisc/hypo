python train_hypo.py \
    --in-dataset CIFAR-10 \
    --id_loc datasets/CIFAR10 \
    --gpu 6 \
    --model resnet18 \
    --loss hypo \
    --epochs 500 \
    --proto_m 0.95 \
    --feat_dim 128 \
    --batch_size 512 \
    --w 2 \
    --cosine




