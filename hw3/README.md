### README

資工三 B10705005 陳思如

#### Environment setup

I use `axolotl` to finetune the Taiwan Llama with Qlora. I created a virtual environment.

-  Setup steps

  ```
  python3 -m venv adlhw3
  cd adlhw3
  source bin/activate
  pip3 install torch==2.0.1
  pip3 install wheel packaging
  git clone https://github.com/OpenAccess-AI-Collective/axolotl
  cd axolotl
  pip3 install -e '.[flash-attn,deepspeed]'
  ```

#### Train

`axolotl` only needs to modify the yml file to set up our counfiguration.

I modified the yml file in the `axolotl/examples/llama-2/qlora.yml` to get the training configuration for finetunig Taiwan-llama2 with qlora.

Here is the yml file:

```
base_model: /tmp2/b10705005/adlhw3/Taiwan-LLM-7B-v2.0-chat
model_type: LlamaForCausalLM
tokenizer_type: LlamaTokenizer
is_llama_derived_model: true

load_in_8bit: false
load_in_4bit: true
strict: false

datasets:
  - path: /tmp2/b10705005/adlhw3/data/trained.json
    type:
      system_prompt: ""
      field_system: system
      format: "[INST] {instruction} [/INST]"
      no_input_format: "[INST] {instruction} [/INST]"
dataset_prepared_path:
val_set_size: 0.05
output_dir: ./qlora-out2

adapter: qlora
lora_model_dir:

sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true

lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
lora_target_linear: true
lora_fan_in_fan_out:

wandb_project:
wandb_entity:
wandb_watch:
wandb_run_id:
wandb_log_model:

gradient_accumulation_steps: 4
micro_batch_size: 2
num_epochs: 4
optimizer: paged_adamw_32bit
lr_scheduler: cosine
learning_rate: 0.0002

train_on_inputs: false
group_by_length: false
bf16: true
fp16: false
tf32: false

gradient_checkpointing: true
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 50
xformers_attention:
flash_attention: true

warmup_steps: 10
eval_steps: 
eval_table_size:
save_steps:
debug:
deepspeed:
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:
  bos_token: "<s>"
  eos_token: "</s>"
  unk_token: "<unk>"
```



After the modification of the yml file, you can start training with the following command.

(run this command in the axolotl directory)

```
accelerate launch --num_processes 1 -m axolotl.cli.train examples/llama-2/qlora.yml
```



#### After

After the training finishes, you will get the adapter checkpoint in the `axolotl/qlora-out2`

