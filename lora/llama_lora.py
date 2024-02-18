"""
llama_lora.py

The class for fine-tuning Llama with LoRA
"""

import torch, transformers
from typing import List

# For Llama
from transformers import LlamaForCausalLM, LlamaTokenizer

# The LLM_Lora base class
from .llm_lora import LLM_Lora

class Llama_Lora(LLM_Lora):
    def __init__(self,
                 base_model: str = "",
                 lora_target_modules: List[str] = ["q_proj", "v_proj"],
                 load_in_8bit: bool = True,
                 cutoff_length: int = 256,
                 ):
        LLM_Lora.__init__(self,
                          base_model = base_model,
                          lora_target_modules = lora_target_modules,
                          load_in_8bit = load_in_8bit,
                          cutoff_length = cutoff_length,
                          )



    def load_base_model(self):
        # Load the model for preparation
        if len(self.base_model) == 0:
            raise ValueError(f"The base_model is {self.base_model}")
        print(f"Load the pre-trained model: {self.base_model}")
        self.model = LlamaForCausalLM.from_pretrained(
            self.base_model,
            load_in_8bit=self.load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        self.tokenizer = LlamaTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Allow batched inference
        
        # print(self.model.parameters)
        # sys.exit()

        # For the generation model
        # Make it model specific
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
