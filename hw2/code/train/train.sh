# training
CUDA_VISIBLE_DEVICES=3 python3 train.py \
    --report_to wandb \
    --model_name_or_path google/mt5-small \
    --train_file trained.json \
    --source_prefix "摘要： " \
    --text_column maintext \
    --summary_column title \
    --output_dir ./4rate/ \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-4 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --num_beams 5 \
    --gradient_accumulation_step 2 \
    --num_train_epochs 5 \
    --num_warmup_steps 300 \