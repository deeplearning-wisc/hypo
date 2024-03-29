python train_cider.py \
    --in-dataset PACS\
    --id_loc datasets/PACS \
    --gpu 1 \
    --model resnet50 \
    --loss cider \
    --epochs 50 \
    --proto_m 0.95 \
    --learning_rate 0.0005 \
    --feat_dim 512 \
    --batch_size 64 \
    --target_domain cartoon \
    --head mlp \
    --w 2 \
    --cosine
