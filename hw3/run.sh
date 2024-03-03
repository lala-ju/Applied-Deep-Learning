# $1 llama
# $2 adapter checkpoint
# $3 input file
# $4 output file
processed_input="process.json"

python3 code/preprocess.py --data $3 --output $processed_input
python3 code/predict.py --base_model_path $1 --peft_path $2 --test_data_path $processed_input --output_data_path $4