import os
import sys
import json
from typing import List, Union

import torch
import transformers
from transformers import GenerationConfig, BitsAndBytesConfig
from datasets import load_dataset

from .utils import identify_label
from .icl import ICL_Demos
from .prompter import Prompter

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)




class LLM_Lora(object):
    """ Generic base class for LoRA fine-tuning
    """
    def __init__(self,
                 base_model: str = "",
                 lora_target_modules: List[str] = [],
                 load_in_8bit: bool = True,
                 cutoff_length: int = 128,
                 task_type: str = 'CAUSAL_LM',
                 ):
        '''
        base_model: the base pre-trained model
        prompt_template_name: the prompt template for instruction tuning/inference
        lora_target_modules: specify the target modules for LoRA (model-dependent)
            This could be an individual function in the future
        load_in_8bit: using quantization to load the pre-trained model
        cutoff_length: the maximal length for input
        '''
        self.base_model = base_model
        self.prompter = None
        self.lora_target_modules = lora_target_modules
        self.cutoff_length = cutoff_length
        self.model = None
        self.tokenizer = None
        self.task_type = task_type
        self.gen_config = None # for generation configuration
        self.train_on_inputs = False
        self.add_eos_token = False
        self.bnb_config = BitsAndBytesConfig(
            load_in_8bit = load_in_8bit,
            )
        

            
    def load_base_model(self):
        print(f"This method needs to be model-specific")


    def load_adapter(self):
        raise NotImplementedError("Has not been implemented yet!")


    def train(self,
              # Data files
              train_file: str = "",
              val_file: str = "",
              data_path: str = "",
              # Output directory
              output_dir: str = "",
              # Training setup
              train_batch_size: int = 32,
              micro_batch_size: int = 4,
              num_epochs: int = 3,
              logging_steps: int = 5,
              warmup_steps: int = 0,
              learning_rate: float = 3e-4,
              # Validation setup
              val_steps: int = 20,
              val_set_size: int = 128,
              val_batch_size: int = 32,
              # LORA
              lora_r: int = 8,
              lora_alpha: int = 16,
              lora_dropout: float = 0.05,
              # Tokenizer
              add_eos_token: bool = True,
              group_by_length: bool = False,
              train_on_inputs: bool = False,
              # Wendb
              wandb_project: str = "",
              wandb_run_name: str = "",
              wandb_watch: str = "",
              wandb_log_model: str = "",
              ):

        print(f"learning rate: {learning_rate}\n")
        # Load the base model
        self.load_base_model()
        # Load the prompter, by default we don't use kshot for traing
        # unless it's MetaICL
        self.prompter = Prompter(kshot=False)

        # Whether training on inputs
        self.train_on_inputs = train_on_inputs
        self.add_eos_token = add_eos_token

        # Set up the configuration
        gradient_accumulation_steps = train_batch_size // micro_batch_size
        
        # =========================================
        # Check if parameter passed or if set within environ
        use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
        )
        # Only overwrite environ if wandb param passed
        if len(wandb_project) > 0:
            os.environ["WANDB_PROJECT"] = wandb_project
        if len(wandb_watch) > 0:
            os.environ["WANDB_WATCH"] = wandb_watch
        if len(wandb_log_model) > 0:
            os.environ["WANDB_LOG_MODEL"] = wandb_log_model
        
        # ==========================================
        # Prepare the model for training
        self.model = prepare_model_for_kbit_training(self.model)
        self.config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=self.task_type,
        )
        self.model = get_peft_model(self.model, self.config)
        self.model.print_trainable_parameters()
        
        # ==========================================
        # Load data
        train_data, val_data = None, None
        if len(data_path) > 0:
            print(f"Load training data from {data_path}")
            train_data = load_dataset(data_path, split="train")
        elif train_file.endswith(".json") or train_file.endswith(".jsonl"):
            print(f"Load training data from {train_file}")
            train_data = load_dataset("json", data_files=train_file, split="train")
        else:
            raise ValueError("Please specify either data_path or trn_data for obtaining training examples")
        
        # Create val set
        if len(data_path) > 0:
            print(f"Load validation data from {data_path}")
            val_data = load_dataset(data_path, split="validation") # Is this a unified split name?
        elif val_file.endswith(".json") or val_file.endswith(".jsonl"):
            # Load the val data from val_file if it is specified
            print(f"Load val data from {val_file}")
            val_data = load_dataset("json", data_files=val_file, split="train")
        elif val_set_size > 0:
            # Split the training set if no val_file
            print(f"Get some val examples from the training set")
            train_val = train_data.train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            train_data = train_val["train"]
            val_data = train_val["test"]
        else:
            print("No validation set")

        
        # Tokenization
        train_data = (
            train_data.shuffle().map(self.generate_and_tokenize_prompt)
        )
        if val_data is not None:
            val_data = (
                val_data.shuffle().map(self.generate_and_tokenize_prompt)
            )

        # Ignore the part of resuming from checkpoints
        
        # ===========================================
        # Trainer
        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                fp16=True,
                logging_steps=logging_steps,
                optim="adamw_torch",
                evaluation_strategy="steps" if (val_data is not None) else "no",
                save_strategy="steps",
                eval_steps=val_steps if (val_data is not None) else None,
                per_device_eval_batch_size=val_batch_size,
                save_steps=val_steps,
                output_dir=output_dir,
                save_total_limit=3,
                save_safetensors=False, # To avoid the bug in SafeTensors loading module
                load_best_model_at_end=True if (val_data is not None) else False,
                ddp_find_unused_parameters=None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name=wandb_run_name if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                self.tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True
            ),
        )
        self.model.config.use_cache = False

        # ===========================================
        # Save model
        # Commented the following block to address the empty file issue
        # old_state_dict = self.model.state_dict
        # self.model.state_dict = (
        #     lambda self, *_, **__: get_peft_model_state_dict(
        #         self, old_state_dict()
        #     )
        # ).__get__(self.model, type(self.model))

        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        trainer.train()

        self.model.save_pretrained(output_dir)


    def predict(self,
                input_file: str = "",
                lora_adapter: str = "",
                # For ICL
                kshot: int = 0,
                demo_file: Union[None, str] = None,
                # Decoding
                do_sample: bool = True,
                top_p: float = 0.8,
                top_k: int = 20,
                num_beams: int = 4,
                temperature: float = 1.0,
                max_new_tokens: int = 128,
                # For classification
                label_set: Union[None, List[str]] = None, # A list of labels
                verbose: bool = False,
                ):
        # Check the input file
        if len(input_file) == 0:
            raise ValueError(f"Empty input file: {input_file}")

        if (kshot > 0) and (demo_file is None):
            raise ValueError(f"In-context learning needs demonstrations, while the demo_file is not specified: {demo_file}")

        # Load the base model
        self.load_base_model()
        self.prompter = Prompter(kshot=kshot, verbose=True)

        # Load the LoRA adapter, if specified
        if len(lora_adapter) == 0:
            print("Only load the base model for inference ...")
        else:
            print(f"Load the LoRA adapter: {lora_adapter}")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_adapter,
                torch_dtype=torch.float16,
            )

        # Generation configuration
        self.gen_config = GenerationConfig(
            do_sample = do_sample,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            num_beams = num_beams,
        )

        # from https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/generate.py#L80
        # not sure whether we need to have this
        # if not self.load_in_8bit:
        #     self.model.half()

        # Q: should this code block be placed before loading the LoRA adapter?
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

            
        # Load test examples
        test_list = []
        with open(input_file) as finput:
            test_list = json.load(finput)


        # Load demo examples for ICL
        if kshot > 0:
            demo_list = []
            with open(demo_file) as fdemo:
                demo_list = json.load(fdemo)
            icl_demos = ICL_Demos(demo_list, kshot)
            
        outputs = []
        for item in test_list:
            demos = None
            if kshot > 0:
                # Construct demos for inference
                demos = icl_demos.generate()

            # Construct a prompt
            res = self._eval(
                instruction=item['instruction'],
                input=item['input'],
                demos = demos,
                max_new_tokens=max_new_tokens,
            )
            
            # print([item for item in res])
            text = [item for item in res][0]

            # print(f"generated response: {text}")
            if label_set is not None:
                text = identify_label(text, label_set)
            
            if verbose:
                try:
                    print(f"MODEL OUTPUT: {text}\nREFERENCE: {item['output']}\n")
                except KeyError:
                    print(f"output: {text}\n")
                print("==========================")

            # Add prediction to the list
            outputs.append(text)
        return outputs
    

    def _eval(self,
              instruction: Union[None, str] = None,
              input: Union[None, str] = None,
              demos: Union[None, List[dict]] = None,
              max_new_tokens: int = 128,
              ):
        # Check the values
        if (instruction is None) or (input is None):
            print(f"Instruction: {instruction}\nInput: {input}")
            raise ValueError("Both instruction and input cannot be None")

        # Format the prompt
        prompt = self.prompter.generate_prompt(
            instruction = instruction,
            input = input,
            demos = demos,
            cutoff_length = self.cutoff_length,
        )
        
        # No truncation, no padding
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        )

        if torch.cuda.is_available():
            input_ids = inputs["input_ids"].to("cuda")
        else:
            input_ids = inputs["input_ids"].to("cpu")

        # Inference
        with torch.no_grad():
            gen_output = self.model.generate(
                input_ids = input_ids,
                generation_config = self.gen_config,
                return_dict_in_generate = True, # ?
                output_scores = True,
                max_new_tokens = max_new_tokens,
            )
        s = gen_output.sequences[0] # Get the output token indices
        output = self.tokenizer.decode(s) # Map to tokens
        yield self.prompter.get_response(output) # Only keep the response part



    def tokenize(self, prompt):
        # No padding, no truncation
        result = self.tokenizer(
            prompt,
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        # Add the EOS token if there is one and add_eos_token is True
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and self.add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result
        


    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
            cutoff_length = self.cutoff_length
        )
        tokenized_full_prompt = self.tokenize(full_prompt)

        # For test
        # output = self.tokenizer.decode(tokenized_full_prompt['input_ids'])
        # print(output)
        # sys.exit()
        return tokenized_full_prompt

