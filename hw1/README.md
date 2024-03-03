## README

B10705005 陳思如

I implemented the training process with the sample code given in the slide. 

All the codes relative to the training process is put in the `train/` directory.

Paragraph selection model training script is saved as `choice.py`. Question answering model training script is saved as `qa.py`

#### 1. Preprocess dataset

Since the sample code has different settings for the training dataset, I have modified the dataset to fit the sample setting with `processchoice.py` and `processqa.py`. These two python script modify the column names, change the content into same format of the sample training set and pop out the unused columns. 

- preprocess the paragraph selection dataset:

  It takes in the original train or valid file and the paragraph context file. It matches the paragraph index in the train/valid file with the context file and put those paragraphs context into the file. It leaves the question as first sentence and the possible paragraphs as second sentence. It saves the preprocessed file as `choice_train.json` and `choice_valid.json`.

  `python3 processchoice.py --filename train.json --context_file context.json`

  `python3 processchoice.py --filename valid.json --context_file context.json`

- preprocess the question answering dataset:

  It takes in the original train or valid file and the paragraph context file. It matches the relative paragraph index in the train/valid file with the context file and put the paragraph content into the file. It only leaves the question and content in the file. It saves the preprocessed file as `qa_train.json` and `qa_valid.json`.

  `python3 processqa.py --filename train.json --context_file context.json`

  `python3 processqa.py --filename valid.json --context_file context.json`

#### 2. Training

The preprocess python script generates the json file that matches the sample dataset format. Then we cant treat these modified dataset as our train and valid file in the training script.

- Paragraph Selection:

  Train the paragraph selection model with below  command and parameters. The trained model will be stored in the output directory `./choice`

  `python3 choice.py --model_name_or_path hfl/chinese-xlnet-base --train_file train/choice_train.json --validation_file choice_valid.json --max_seq_length 512 --per_device_train_batch_size 1 --learning_rate 3e-5 --num_train_epochs 1 --gradient_accumulation_steps 1 --num_warmup_steps 100 --lr_scheduler_type polynomial --output_dir ./choice`

- Question Answering:

  Train the question answering model with this command and parameters. The trained model will be stored in the output directory `./qa`

  `python3 qa.py --model_name_or_path hfl/chinese-roberta-wwm-ext --train_file train/qa_train.json --validation_file qa_valid.json --max_seq_length 512 --per_device_train_batch_size 1 --learning_rate 3e-5 --num_train_epochs 3 --gradient_accumulation_steps 1 --num_warmup_steps 100 --lr_scheduler_type polynomial --output_dir ./qa`

#### 3. Implement the trained model

After two scripts has finished the training, we will get two models for paragraph selection and question answering. Then we can loaded them to predict our test files. But before any file sent into these two processes, we have to make sure the dataset has been preprocessed. 

Requirement for dataset:

- paragraph selection

  Contains only `id`, `sent1`, `sent2`, `concat sentences after sent2` x 4.

  In our case, the `id` stays the same. `sent1` is our question. `sent2` is empty since the whole paragraph is our choice. `ending0` `ending1` `ending2` `ending3` these are the concat sentences after sent2 which is our four paragraphs choices. Other columns is popped.

- question answering

  Contains only `id`, `question`, `context`.

  In our case, the `id` and `question` stays the same. `context` is the relevant paragraph content that we have found through the paragraph selection model. Then other columns is popped.

#### 4. Test and predict

Then we move to our `test/` directory to run out test file with our trained model.

1. Preprocess the dataset first, put dataset file into the `processchoice.py`

   `python3 processchoice.py --filename path\to\test.json --context_file path\to\context.json`

2. Predict the paragraph selection result with `choicepredict.py`

   `python3 choicepredict.py --model_name_or_path path/to/choice_model_dir --config_name path/to/choice_model_dir --tokenizer_name path/to/choice_model_dir --trust_remote_code True --test_file choice_test.json --max_seq_length 512 --gradient_accumulation_steps 1  --output_dir $PWD`

3. The prediction of the choice will generate one json file contains the predicted label of the choice. We need to process the dataset to make it to the format for the qa model prediction with `postprocesschoice.py`

   `jq -s . choiceresult_test.json > choiceresult.json`

   `python3 postprocesschoice.py --filename choiceresult.json --original_file path/to/test.json --context_file path/to/context.json`

4. Predict the question answering result with `qapredict.py`

   `python3 qapredict.py --model_name_or_path path/to/qa_model_dir --config_name path/to/qa_model_dir --tokenizer_name path/to/qa_model_dir --trust_remote_code True --test_file qa_test.json --max_seq_length 512 --gradient_accumulation_steps 1  --output_dir $PWD`

5. The prediction of the question answering model generate the json file containing answer. We need to transform it into the csv file for submission with `postprocessqa.py`

   `jq -s . qaresult_test.json > qaresult.json`

   `python3 postprocessqa.py --filename qaresult.json --final_filename path\to\submission.csv`