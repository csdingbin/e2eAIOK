#!/bin/bash
cd "$(dirname "$0")"
set -e

cd ../

budget_model_size=1e6
max_layers=18
population_size=512
evolution_max_iter=10000  # we suggest evolution_max_iter=480000 for


save_dir=../save_dir/DENAS_JC_0_5_cifar_params1M
mkdir -p ${save_dir}

echo "SuperConvK3BNRELU(3,8,1,1)SuperResK3K3(8,16,1,8,1)SuperResK3K3(16,32,2,16,1)SuperResK3K3(32,64,2,32,1)SuperResK3K3(64,64,2,32,1)SuperConvK1BNRELU(64,128,1,1)" \
> ${save_dir}/init_plainnet.txt

# python -u evolution_search.py \
#   --zero_shot_score DENAS_JC \
#   --search_space SearchSpace/search_space_XXBL.py \
#   --budget_model_size ${budget_model_size} \
#   --max_layers ${max_layers} \
#   --batch_size 64 \
#   --input_image_size 32 \
#   --plainnet_struct_txt ${save_dir}/init_plainnet.txt \
#   --num_classes 100 \
#   --evolution_max_iter ${evolution_max_iter} \
#   --population_size ${population_size} \
#   --save_dir ${save_dir} \
#   --belta 0.7


# python -u analyze_model.py \
#   --input_image_size 32 \
#   --num_classes 100 \
#   --arch Masternet.py:MasterNet \
#   --plainnet_struct_txt ${save_dir}/best_structure.txt

python train_image_classification.py --dataset cifar10 --num_classes 10 \
  --dist_mode cpu \
  --input_image_size 32 --epochs 1440 --warmup 5 \
  --optimizer sgd --bn_momentum 0.01 --wd 5e-4 --nesterov --weight_init custom \
  --label_smoothing --random_erase --mixup --auto_augment \
  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt \
  --batch_size_per_gpu 64 \
  --save_dir ${save_dir}/cifar10_1440epochs



# python train_image_classification.py --dataset cifar100 --num_classes 100 \
#   --dist_mode cpu \
#   --input_image_size 32 --epochs 1440 --warmup 5 \
#   --optimizer sgd --bn_momentum 0.01 --wd 5e-4 --nesterov --weight_init custom \
#   --label_smoothing --random_erase --mixup --auto_augment \
#   --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
#   --arch Masternet.py:MasterNet \
#   --plainnet_struct_txt ${save_dir}/best_structure.txt \
#   --batch_size_per_gpu 64 \
#   --save_dir ${save_dir}/cifar100_1440epochs
