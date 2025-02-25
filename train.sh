export CUDA_VISIBLE_DEVICES=0
python trainer.py \
    --model clip_vit_b_16 --input_size 224 --reduction 8 --truncation 4 \
    --dataset qnrf --batch_size 16 --amp \
    --num_crops 2 --sliding_window --window_size 224 --stride 224 --warmup_lr 1e-3 \
    --count_loss dmcount
