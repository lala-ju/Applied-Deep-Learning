CUDA_VISIBLE_DEVICES=5 python result.py \
    --model_name_or_path 4rate \
    --test_file publiced.json \
    --max_source_length 1024 \
    --max_target_length 128 \
    --source_prefix "摘要： "\
    --text_column maintext \
    --trust_remote_code True\
    --output_file ./publicsubmission/sub.jsonl \
    --beam 5\