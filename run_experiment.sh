#!/bin/bash
# ./run_experiment --dataset --num_classes --tree_max_depth --batch_size  --tree_n_cl(how many clusters at each depth) --model --cuda_device_id 
#CUDA_VISIBLE_DEVICE=$7 python train_baseline.py --dataset $1 --num_classes $2  --epochs 50 --model $6
CUDA_VISIBLE_DEVICE=$7 python construct_dict.py  --save_activations --model $6 --batch_size $4 --layer 1 --dataset $1  --pretrained_model "./models/${6}_${1}_baseline.pth" --max_depth $3 --n_cl $5 --num_classes $2
CUDA_VISIBLE_DEVICE=$7 python train_quantized.py --layer 1 --model $6 --dataset $1 --pretrained_model "./models/${6}_${1}_baseline.pth" --max_depth $3 --n_cl $5  --num_classes $2 --epochs 10
CUDA_VISIBLE_DEVICE=$7 python construct_dict.py   --save_activations --model $6 --batch_size $4 --layer 2 --dataset $1  --pretrained_model "./models/${6}_${1}_baseline.pth" --max_depth $3 --n_cl $5 --num_classe $2
CUDA_VISIBLE_DEVICE=$7 python train_quantized.py --layer 2 --model $6 --dataset $1 --pretrained_model "./models/${6}_${1}_ql_layer1_depth_${3}_n_cl_${5}.pth" --max_depth $3 --n_cl $5   --num_class $2 --epochs 10
if [ "$1" = "german" ]
then
echo "Additional Layer for German Road Sign"
CUDA_VISIBLE_DEVICE=$7 python construct_dict.py  --save_activations --model $6  --batch_size 300 --layer 3 --dataset $1  --pretrained_model "./models/${6}_${1}_baseline.pth" --max_depth $3 --n_cl $5 --num_classe $2
CUDA_VISIBLE_DEVICE=$7 python train_quantized.py --layer 3 --model $6 --dataset $1 --pretrained_model "./models/${6}_${1}_ql_layer2_depth_${3}_n_cl_${5}.pth" --max_depth $3 --n_cl $5  --num_class $2 --epochs 10
else
echo "Not German. Done"
fi
if [ "$6" = "resnet" ]
then
echo "Additional Layers for ResNet"
CUDA_VISIBLE_DEVICE=$7 python construct_dict.py  --save_activations --model $6  --batch_size 500 --layer 3 --dataset $1  --pretrained_model "./models/${6}_${1}_baseline.pth" --max_depth $3 --n_cl $5 --num_classe $2
CUDA_VISIBLE_DEVICE=$7 python train_quantized.py --layer 3 --model $6 --dataset $1 --pretrained_model "./models/${6}_${1}_ql_layer2_depth_${3}_n_cl_${5}.pth" --max_depth $3 --n_cl $5  --num_class $2 --epochs 10
CUDA_VISIBLE_DEVICE=$7 python construct_dict.py  --save_activations --model $6  --batch_size 500 --layer 4 --dataset $1  --pretrained_model "./models/${6}_${1}_baseline.pth" --max_depth $3 --n_cl $5 --num_classe $2
CUDA_VISIBLE_DEVICE=$7 python train_quantized.py --layer 4 --model $6 --dataset $1 --pretrained_model "./models/${6}_${1}_ql_layer3_depth_${3}_n_cl_${5}.pth" --max_depth $3 --n_cl $5  --num_class $2 --epochs 10
else
echo "Not ResNet. Done"
fi

