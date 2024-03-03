#S1 input
#$2 output
testjson="testdata.json"
model="model"
# preprocess input file
python3 preprocess.py --input $1 --output $testjson
# run prediction
python3 result.py \
    --model_name_or_path $model \
    --test_file $testjson \
    --max_source_length 1024 \
    --max_target_length 128 \
    --source_prefix "摘要： "\
    --text_column maintext \
    --trust_remote_code True\
    --output_file $2 \
    --beam 5\