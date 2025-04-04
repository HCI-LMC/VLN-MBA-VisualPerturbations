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
name=${name}-seed.${seed}
name=${name}-init.aug.45k

outdir=../Out/R2R/finetune/DUET-clip-VlMS-VlOG-VgOG-VgMS-t

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert      

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200

      --batch_size 8
      --lr 1e-5
      --iters 50000
      --log_every 1000
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --depth_feat_size ${dft_dim}
      --angle_feat_size 4

      --ml_weight 0.2   

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0."

python -c 'print(" ")'
python -c 'print("1------------------------------------------------------------------------1")'
python -c 'print("1--------------    r2r  预训练  微调训练 原始的duet-clip    VlMS-VlOG-VgOG-VgMS       -----------------1")'
python -c 'print("1------------------------------- ---- -----------------------------------1")'
python -c 'print(" ")'
# train
#CUDA_VISIBLE_DEVICES='7' /home/zhangxuesong/.conda/envs/vlnduet/bin/python r2r/main_nav.py $flag  \
#      --tokenizer bert \
#      --bert_ckpt_file '/data/zxs/Matterport3DSimulator/VLN-DUET/VLN-DUET-clip-new-original/Out/R2R/pretrain/DUET-clip-original/ckpts/model_step_100000.pt' \
##      --eval_first
#
# test
CUDA_VISIBLE_DEVICES='6' /home/zhangxuesong/.conda/envs/vlnduet/bin/python r2r/main_nav.py $flag  \
      --tokenizer bert \
      --resume_file /data2/zxs/Backup/VLN-DUET/4-DUET-Clip-VlMS-VlOG-VgOG-VgMS/Out/R2R/finetune/DUET-clip-VlMS-VlOG-VgOG-VgMS/ckpts/best_val_unseen \
      --test --submit