from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    phrases = instruction.split('\n')
    
    instr = ''
    input = ''
    newline = '\n'
    for phrase in phrases:
        if '翻譯' in phrase or '怎麼說' in phrase:
            instr += phrase
        elif '答案' not in phrase:
            input += phrase
            
    if '文言' in instr:
        instr = '翻譯成文言文。'
    elif '古代' in instr:
        instr = '翻譯成中國古代的話。'
    elif '古文' in instr:
        instr = '翻譯成古文。'
    elif '現代' in instr:
        instr = '翻譯成現代文。'
    elif '白話' in instr:
        instr = '翻譯成白話文。'
    return f'請將下列文字{instr}{input}答案：'

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    return BitsAndBytesConfig(
        bnb_4bit_compute_dtype='bfloat16',
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
        llm_int8_skip_modules=None,
        llm_int8_threshold=6.0,
        load_in_4bit=True,
        load_in_8bit=False,
        quant_method='bitsandbytes'
    )
