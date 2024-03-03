### README

#### How to train and predict

1. preprocess the train file and validation file from `jsonl` to `json`

   ``` 
   python preprocess --input [path/to/jsonl] --output [path/to/json]
   ```

2. sent the train file and validation file into the transformer to fine tune the mT5 model

   ```
   python train.py \
       --model_name_or_path google/mt5-small \
       --train_file [/path/to/train/json/file] \
       --validation_file [/path/to/validation/json/file] \
       --source_prefix "摘要： " \
       --text_column maintext \
       --summary_column title \
       --output_dir [dir/path/to/save/model] \
       --per_device_train_batch_size 2 \
       --learning_rate 1e-4 \
       --max_source_length 1024 \
       --max_target_length 128 \
       --num_beams 5 \
       --gradient_accumulation_step 2 \
       --num_train_epochs 5 \
       --num_warmup_steps 300 \
   ```

3. Then we get the model at `[dir/path/to/save/model]`. We can predict the result with this model.

4. We also need to preprocess the test file.

   ```
   python preprocess --input [path/to/jsonl] --output [path/to/json]
   ```

5. After the preprocess, we can sent the file into prediction and get the summarization out.

   ```
   python result.py \
       --model_name_or_path [dir/path/to/save/model]\
       --test_file [/path/to/test/json/file] \
       --max_source_length 1024 \
       --max_target_length 128 \
       --source_prefix "摘要： "\
       --text_column maintext \
       --trust_remote_code True\
       --output_file [path/to/submission/jsonl/file]\
       --beam 5\
   ```

   