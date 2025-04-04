DATA_ROOT=/data/public_datasets/VLN/zxs/datasets
#DATA_ROOT=/dev/shm/zxs/datasets
train_alg=dagger

features=clip
ft_dim=768
dft_dim=1000
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

name=${train_alg}-${features}
name=${name}-seed.${seed} #-${ngpus}gpus

outdir=../Out/REVERIE/finetune/DUET-clip-VlMS-VlOG-VgOG-VgMS-t25

flag="--root_dir ${DATA_ROOT}
      --dataset reverie
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert

      --enc_full_graph
      --graph_sprels
      --fusion dynamic
      --multi_endpoints

      --dagger_sample sample

      --train_alg ${train_alg}

      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2

      --max_action_len 15
      --max_instr_len 200
      --max_objects 20

      --batch_size 8
      --lr 1e-5
      --iters 25000
      --log_every 1000
      --optim adamW

      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --depth_feat_size ${dft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5

      --gamma 0."

python -c 'print(" ")'
python -c 'print("1------------------------------------------------------------------------1")'
python -c 'print("1--------------      预训练100k  微调训练 原始的duet-clip   VlMS-VlOG-VgOG-VgMS     -----------------1")'
python -c 'print("1------------------------------- ---- -----------------------------------1")'
python -c 'print(" ")'
## train
CUDA_VISIBLE_DEVICES='2' /home/zhangxuesong/.conda/envs/vlnduet/bin/python reverie/main_nav_obj.py $flag  \
      --tokenizer bert \
      --bert_ckpt_file '/data/zxs/Matterport3DSimulator/VLN-DUET/VLN-DUET-clip-new-original/part-Out/REVERIE/pretrain/DUET_CLIP-original/ckpts/model_step_100000.pt' \
      --eval_first

## test
#CUDA_VISIBLE_DEVICES='6' /home/zhangxuesong/.conda/envs/vlnduet/bin/python reverie/main_nav_obj.py $flag  \
#      --tokenizer bert \
#      --resume_file /data/zxs/Matterport3DSimulator/VLN-DUET/4-DUET-Clip-VlMS-VlOG-VgOG-VgMS/Out/REVERIE/finetune/DUET-clip-VlMS-VlOG-VgOG-VgMS-t/ckpts/best_val_unseen \
#      --test --submit