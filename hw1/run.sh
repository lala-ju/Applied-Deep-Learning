#model dir name
choicemodel='choice'
qamodel='qa'
# $1 context $2 test $3 sub.csv
python3 processchoice.py --context_file $1 --filename $2 
# choice_test
python3 choicepredict.py --model_name_or_path $choicemodel --config_name $choicemodel --tokenizer_name $choicemodel --trust_remote_code True --test_file choice_test.json --max_seq_length 512 --gradient_accumulation_steps 1  --output_dir $PWD
# python choicepredict.py 
# choiceresult_test
jq -s . choiceresult_test.json > choiceresult.json
# choiceresult
python3 postprocesschoice.py --filename choiceresult.json --original_file $2 --context_file $1
# python postprocesschoice.py --filename choiceresult.json --original_file $2 --context_file $1
# qa_test
python3 qapredict.py --model_name_or_path $qamodel --config_name $qamodel --tokenizer_name $qamodel --trust_remote_code True --test_file qa_test.json --max_seq_length 512 --gradient_accumulation_steps 1  --output_dir $PWD
# python qapredict.py
# qaresult_test
jq -s . qaresult_test.json > qaresult.json
#qaresult 
python3 postprocessqa.py --filename qaresult.json --final_filename $3

rm *.json