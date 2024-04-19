# for cartoon, running script with 'learning_rate' 0.0005, 'batch_size' 32, 'w' 4;
# for photo, running script with 'learning_rate' 0.0001, 'batch_size' 32, 'w' 1;
# for sketch, running script with 'learning_rate' 0.002, 'batch_size' 32, 'w' 2;
# for art_painting, running script with 'learning_rate' 0.0005, 'batch_size' 32, 'w' 1;
python train_hypo.py \
    --in-dataset PACS\
    --id_loc datasets/PACS \
    --gpu 1 \
    --model resnet50 \
    --loss hypo \
    --epochs 50 \
    --proto_m 0.95 \
    --learning_rate 0.0005 \
    --feat_dim 512 \
    --batch_size 32 \
    --target_domain cartoon \
    --head mlp \
    --w 4 \
    --cosine


