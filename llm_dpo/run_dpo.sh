torchrun --nproc_per_node=8 --master_port=60000 dpo.py \
    --max_prompt_length 512 \
    --model_name_or_path ./models/llama2/hh_sft \
    --train_path ./datasets/hh_data/hh_rm_train.json \
    --val_path ./datasets/hh_data/hh_rm_val.json \
    --output_path ./models/dpo_llama2/adaptive_temp_lr5e-5_rho0.1_beta0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --beta 0.1 \
    --rho 0.1 \
    --tau_max 5 \
    --tau_min 0.1 \
    --eval_steps 1000 \
    --num_train_epochs 1 \
    --loss_type adaptive_temp \