"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
import os
from typing import Union, List


class Prompter(object):

    def __init__(self,
                 kshot: int = 0,
                 verbose: bool = False,
                 ):
        self.verbose = verbose
        self.kshot = kshot
        # Prompt template
        print(os.getcwd())
        file_name = None
        if kshot > 0:
            file_name = osp.join("lora/templates", "kshot.json")
        else:
            file_name = osp.join("lora/templates", "base.json")
            
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")

        with open(file_name) as fp:
            self.template = json.load(fp)
        
        if self.verbose:
            print(
                f"Using prompt template {file_name}: {self.template['description']}"
            )


    
    def generate_prompt(
            self,
            instruction: str,
            input: Union[None, str] = None,
            output: Union[None, str] = None,
            demos: Union[None, List[dict]] = None,
            cutoff_length: int = 128,
    ) -> str:
        if input is None:
            raise KeyError(f"No input value for the task")

        # truncate it
        tokens = input.split()
        if len(tokens) > cutoff_length:
            input = " ".join(tokens[:cutoff_length])
        
        # 
        if self.kshot == 0:
            res = self.template["prompt_input"].format(
                instruction = instruction,
                input = input,
            )
        elif self.kshot > 0:
            # In-context learning
            if demos is None:
                raise ValueError("No demostrations for ICL")
            
            # Instruction part
            res = self.template["instruction"].format(
                instruction = instruction,
            )
            
            # Demonstration part
            for item in demos:
                item_input = item["input"]
                tokens = item_input.split()
                if len(tokens) > cutoff_length:
                    item_input = " ".join(tokens[:cutoff_length])
                demo_text = self.template["demo_part"].format(
                    input = item_input,
                    output = item["output"],
                )
                res = f"{res}{demo_text}"
            # Query (input) part
            query_text = self.template["query_part"].format(
                input=input,
            )
            res = f"{res}{query_text}"
        else:
            raise ValueError(f"The number of examples for ICL cannot be negative: {kshot}")

        # print(f"Constructed result: {res}")
        if output:
            res = f"{res}{output}" # Pair the input with output
        if self.verbose:
            print(res)
        return res

    

    def get_response(self, output: str) -> str:
        if self.template["response_split"] in output:
            # Get the last chuck, prepare for in-context learning
            return output.split(self.template["response_split"])[-1].strip()
        else:
            raise ValueError(f"Unrecognized string: {output}")
