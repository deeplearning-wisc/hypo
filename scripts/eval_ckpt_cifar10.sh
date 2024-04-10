ID_DATASET=CIFAR-10 

ID_LOC=/nobackup2/yf/datasets
OOD_LOC=/nobackup2/yf/datasets

CKPT_LOC=/nobackup2/yf/checkpoints/CIFAR-10
CKPT_NAME=ckpt_hypo_resnet18_cifar10

CKPT_LOC=/nobackup2/yf/checkpoints/hypo_cr/CIFAR-10/09_04_20:02_hypo_resnet18_lr_0.0005_cosine_True_bsz_512_head_mlp_wd_2.0_500_128_trial_0_temp_0.1_CIFAR-10_pm_0.95
CKPT_NAME=checkpoint_max

for cortype in 'gaussian_noise' 'zoom_blur' 'impulse_noise' 'defocus_blur' 'snow' 'brightness' 'contrast' 'elastic_transform' 'fog' 'frost' 'gaussian_blur' 'glass_blur' 'jpeg_compression' 'motion_blur' 'pixelate' 'saturate' 'shot_noise' 'spatter' 'speckle_noise'
do
    python eval_hypo.py --model resnet18 --head mlp --gpu 0 --cortype=$cortype --in-dataset ${ID_DATASET} --id_loc ${ID_LOC} --ood_loc ${OOD_LOC} --ckpt_name ${CKPT_NAME} --ckpt_loc ${CKPT_LOC}
done
