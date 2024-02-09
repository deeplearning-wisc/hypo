NAME=$1
ID_DATASET=CIFAR-10 
ID_LOC=datasets/CIFAR10
OOD_LOC=datasets/small_OOD_dataset


for cortype in 'gaussian_noise' 'zoom_blur' 'impulse_noise' 'defocus_blur' 'snow' 'brightness' 'contrast' 'elastic_transform' 'fog' 'frost' 'gaussian_blur' 'glass_blur' 'jpeg_compression' 'motion_blur' 'pixelate' 'saturate' 'shot_noise' 'spatter' 'speckle_noise'
do
    python eval_hypo_cifar.py --model resnet18 --head mlp --gpu 0 --cortype=$cortype
done
