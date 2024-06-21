#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------------------
#                                           IMPORT
# --------------------------------------------------------------------------------------------
import requests

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from peft import PeftModel   

import numpy as np
from loguru import logger
import wandb
import pandas as pd
from decouple import config
import os
import yaml

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


os.environ["REPLICATE_API_TOKEN"] = config('REPLICATE_API_TOKEN')

# Important: import replicate after setting the environment variable.
import replicate

# ---------------------------------------- Setups --------------------------------------
OPENAI_KEY = config('OPENAI_KEY')
OPENAI_ORG = config('OPENAI_ORG')

# ----------------------------------------- Models ----------------------------------------
class OpenAIModel(object):
    def __init__(self, model_name="gpt-3.5-turbo-0613", configs=None):

        self.is_min_temperature = configs['is_min_temperature']
        self.is_max_temperature = configs['is_max_temperature']
        if self.is_min_temperature:
            self.openai_temperature_ablation = configs['min_openai_temperature_ablation']
        elif self.is_max_temperature:
            self.openai_temperature_ablation = configs['max_openai_temperature_ablation']
        
        if self.is_min_temperature or self.is_max_temperature:
            self.chat_llm = ChatOpenAI(
                openai_api_key=OPENAI_KEY, 
                model_name=model_name,
                temperature=self.openai_temperature_ablation,
                # max_tokens=250,
            )
        else:
            self.chat_llm = ChatOpenAI(
                openai_api_key=OPENAI_KEY, 
                model_name=model_name,
            )

        system_template = ""
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        
        human_template = "{response_length}{context}{instruction}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        self.chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


    def generate(self, instruction, task, context=None, response_length=None):
        
        if context and not pd.isnull(context):
            new_context = "Context: " + context + "\n"
            logger.debug(f"Context: {new_context}")
        else:
            new_context = ""
            logger.debug(f"Context is empty")


        if response_length:
            new_response_length = f"Response length: {response_length} characters\n" 
            logger.debug(f"Response length: {new_response_length} characters")
        else:
            new_response_length = ""
            logger.debug(f"Response length is empty")

        logger.debug(f"Instruction: {instruction}")

        try:
            chain = LLMChain(llm=self.chat_llm, prompt=self.chat_prompt)
            gpt_response = chain.run(instruction=instruction, context=new_context, response_length=new_response_length)
            
            logger.debug(f"Response: {gpt_response}")

            return gpt_response.strip()
        
        except Exception as e:
            logger.error(f"Error: {e}")
            return ""



class ReplicateModel(object):
    def __init__(self, model_name="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3", configs=None):
        
        self.model_name  =  model_name


    def generate(self, instruction, task=None, context=None, response_length=None):
        
        
        
        # new_task = "Task:" + task + "\n"
        if context and not pd.isnull(context):
            new_context = "Context: " + context + "\n"
            logger.debug(f"Context: {new_context}")
        else:
            new_context = ""
            logger.debug(f"Context is empty")

        if response_length:
            new_response_length = f"Response length: {response_length} characters\n" 
            logger.debug(f"Response length: {new_response_length} characters")
        else:
            new_response_length = ""
            logger.debug(f"Response length is empty")

        logger.debug(f"Instruction: {instruction}")

        try:
            input_str = f"{new_response_length}{new_context}{instruction}"
            output = replicate.run(
                self.model_name,
                input={"prompt": input_str}
            )

            response = "".join(output)          
            logger.debug(f"Response: {response}")

            return response.strip()
        
        except Exception as e:
            logger.error(f"Error: {e}")
            return ""



class HFInferenceModel(object):

    def __init__(self, model_name="meta-llama/Llama-2-70b-chat-hf", temperature=1, return_full_text=False, configs=None):
        
        self.is_min_temperature = configs['is_min_temperature']
        self.is_max_temperature = configs['is_max_temperature']

        if self.is_min_temperature:
            self.hf_temperature_ablation = configs['min_hf_temperature_ablation']
            self.hf_do_sample_ablation = configs['min_hf_do_sample_ablation']
        elif self.is_max_temperature:
            self.hf_temperature_ablation = configs['max_hf_temperature_ablation']
            self.hf_do_sample_ablation = configs['max_hf_do_sample_ablation']

        self.model_name  =  model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.header = {"Authorization": f"Bearer {config('HF_API_TOKEN')}"}

        # Check possible parameters here: https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
        if self.is_min_temperature or self.is_max_temperature:
            self.parameters = {
                "temperature": self.hf_temperature_ablation, #! Before ablation: removed
                "do_sample": self.hf_do_sample_ablation, #! Before ablation: removed
                "return_full_text": return_full_text,
            }
        else:
            self.parameters = {
                "return_full_text": return_full_text,
                "do_sample": True,
            }

        
    def generate(self, instruction, task=None, context=None,  response_length=None):
        
    
        # new_task = "Task:" + task + "\n"
        if context and not pd.isnull(context):
            new_context = "Context: " + context + "\n"
            logger.debug(f"Context: {new_context}")
        else:
            new_context = ""
            logger.debug(f"Context is empty")

        if response_length:
            new_response_length = f"Response length: {response_length} characters\n" 
            logger.debug(f"Response length: {new_response_length} characters")
        else:
            new_response_length = ""
            logger.debug(f"Response length is empty")

        logger.debug(f"Instruction: {instruction}")

        try:
            input_str = f"{new_response_length}{new_context}{instruction}"
            logger.debug(f"Input: {input_str}")
            # Query the HF Inference API
            payload = {"inputs": input_str, "parameters": self.parameters}
            logger.debug(f"Payload: {payload}")
            api_response = requests.post(self.api_url, headers=self.header, json=payload)
            logger.debug(f"API Response: {api_response}")
            logger.debug(f"API Response JSON: {api_response.json()}")
            response = api_response.json()[0]["generated_text"]
                       
            logger.debug(f"Response: {response}")
            

            return response.strip()
        
        except Exception as e:
            logger.error(f"Error: {e}")
            return ""
        

class HFModel():
    def __init__(self, model_name, adapter_name=None, generator_type=None, configs=None):

        self.is_min_temperature = configs['is_min_temperature']
        self.is_max_temperature = configs['is_max_temperature']

        if self.is_min_temperature:
            self.hf_temperature_ablation = configs['min_hf_temperature_ablation']
            self.hf_do_sample_ablation = configs['min_hf_do_sample_ablation']
        elif self.is_max_temperature:
            self.hf_temperature_ablation = configs['max_hf_temperature_ablation']
            self.hf_do_sample_ablation = configs['max_hf_do_sample_ablation']


        self.model_name = model_name
        self.tokenizer_name = model_name
        self.adapter_name = adapter_name
        self.generator_type = generator_type

        if "falcon" in self.model_name.lower():
            self.max_length = 2048
        elif "llama" in self.model_name.lower():
            self.max_length = 4096
        else:
            raise ValueError("Cannot set max_length, HFModel name not supported")
    
    def _load_pipeline_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )


    def _load_peft_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory= {i: '24000MB' for i in range(torch.cuda.device_count())},
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(self.model, self.adapter_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _load_causallm_model(self):
        raise NotImplementedError("Load causal LM model not implemented yet")

    def load_model(self):
        if self.generator_type == "causallm":
            self._load_causallm_model()

        if self.generator_type == "peft":
            self._load_peft_model()

        elif self.generator_type == "pipeline":
            self._load_pipeline_model()

        else:
           raise ValueError("Generator type not supported")
        

    def _format_text(self, input_text, task, context=None, response_length=None):
        if self.model_name in ["tiiuae/falcon-7b-instruct", "meta-llama/Llama-2-7b-chat-hf"]:
            
            if context and not pd.isnull(context):
                formatted_text =  f"Context: {context}\n{input_text}"

                if response_length:
                    formatted_text = f"Response length: {response_length} characters\n" + formatted_text

                return formatted_text
            else:
                formatted_text =  f"{input_text}"
                
                if response_length:
                    formatted_text = f"Response length: {response_length} characters\n" + formatted_text

                return formatted_text

        elif self.adapter_name in ["timdettmers/guanaco-7b", "jco2/falcon-7b-dolly-peft-stp-10k"]:
            if context and not pd.isnull(context):
                formatted_text =  f"Context: {context}\n### Human: {input_text} ### Assistant:"
                
                if response_length:
                    formatted_text = f"Response length: {response_length} characters\n" + formatted_text   

                return formatted_text
            else:
                formatted_text =  f"### Human: {input_text} ### Assistant:"
                
                if response_length:
                    formatted_text = f"Response length: {response_length} characters\n" + formatted_text

                return formatted_text
        
        else:
            raise ValueError("Model name not supported")


    def _generate_pipeline(self, prompt):
        try:
            
            if self.is_max_temperature or self.is_min_temperature:
                
                sequences = self.pipeline(
                    prompt,
                    max_length= self.max_length,
                    do_sample=self.hf_do_sample_ablation, #! Before ablation: True
                    # top_k=10,
                    num_return_sequences=1,
                    # eos_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False,
                    temperature=self.hf_temperature_ablation, #! Before ablation: removed
                )
            else:
                sequences = self.pipeline(
                    prompt,
                    max_length= self.max_length,
                    do_sample=True,
                    # top_k=10,
                    num_return_sequences=1,
                    # eos_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False,
                )

            # Since num_return_sequences=1, we only have one sequence ([0])
            return sequences[0]["generated_text"].strip()
        except Exception as e:
            logger.error(f"Error: {e}")
            return ""
    

    def _generate_peft(self, prompt):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
            outputs = self.model.generate( inputs=inputs.input_ids, max_length=self.max_length)
            
            # outputs[0] --> Get input and output text, outputs[:,inputs.input_ids.shape[-1]:][0] --> Get output text only
            decoded_output = self.tokenizer.decode(outputs[:,inputs.input_ids.shape[-1]:][0], skip_special_tokens=True)

            out = decoded_output.split("### Human:")[0].split("Human:")[0].strip()
            return out
        except Exception as e:
            logger.error(f"Error: {e}")
            return ""

    
    def _generate_causallm(self, prompt):
        raise NotImplementedError("Generate causal LM not implemented yet")
    
    def generate(self, input_text, task, context=None, response_length=None):
        prompt = self._format_text(input_text, task, context, response_length)

        if self.generator_type == "pipeline":
            return self._generate_pipeline(prompt)
        elif self.generator_type == "peft":
            return self._generate_peft(prompt)
        elif self.generator_type == "causallm":
            return self._generate_causallm(prompt)
        else:
            raise ValueError("Generator type not supported")
   

if __name__ == "__main__":

    with open('../configs/configs.yaml') as f:
        configs = yaml.safe_load(f)

    # Assert if more than one is true
    assert sum([configs['is_min_temperature'], configs['is_max_temperature']]) <= 1, "Only one ablation experiment can be true"

    IS_ABLATION_LENGTH_TEST = False
    if IS_ABLATION_LENGTH_TEST:
        print("********** ABLATION LENGTH TEST **********")

    d = [
        {"test_context": "", "test_input": "What are the best selling albums of all time? Give me the list with the name of the artist and the name of the album", "test_task": "Brainstorming", "test_response": "Best selling albums of all time:\n\n- Michael Jackson: Thriller\n- Eagles: Their Greatest Hits (1971–1975)\n- Pink Floyd: The Dark Side of the Moon\n- Whitney Houston / various artists: The Bodyguard\n- Bee Gees / Various artists: Saturday Night Fever\n- Fleetwood Mac: Rumours\n- Eagles: Hotel California\n- Shania Twain: Come On Over\n- Meat Loaf: Bat Out of Hell\n- AC/DC: Back in Black"},
        {"test_context": "Komorida was born in Kumamoto Prefecture on July 10, 1981. After graduating from high school, he joined the J1 League club Avispa Fukuoka in 2000. Although he debuted as a midfielder in 2001, he did not play much and the club was relegated to the J2 League at the end of the 2001 season. In 2002, he moved to the J2 club Oita Trinita. He became a regular player as a defensive midfielder and the club won the championship in 2002 and was promoted in 2003. He played many matches until 2005. In September 2005, he moved to the J2 club Montedio Yamagata. In 2006, he moved to the J2 club Vissel Kobe. Although he became a regular player as a defensive midfielder, his gradually was played less during the summer. In 2007, he moved to the Japan Football League club Rosso Kumamoto (later Roasso Kumamoto) based in his local region. He played as a regular player and the club was promoted to J2 in 2008. Although he did not play as much, he still played in many matches. In 2010, he moved to Indonesia and joined Persela Lamongan. In July 2010, he returned to Japan and joined the J2 club Giravanz Kitakyushu. He played often as a defensive midfielder and center back until 2012 when he retired.", "test_input": "When was Komorida born?", "test_task": "Open-domain question answering", "test_response": "Tomoaki Komorida was born on July 10,1981."},
    ]

    models = [
        # {"model_name": "meta-llama/Llama-2-7b-chat-hf", "adapter_name": None, "generator_type": "pipeline", "generator": "hf_model"},
        # {"model_name": "tiiuae/falcon-7b-instruct", "adapter_name": None, "generator_type": "pipeline", "generator": "hf_model"},
        # {"model_name": "huggyllama/llama-7b", "adapter_name": 'timdettmers/guanaco-7b', "generator_type": "peft", "generator": "hf_model"},
        # {"model_name": "tiiuae/falcon-7b", "adapter_name": 'jco2/falcon-7b-dolly-peft-stp-10k', "generator_type": "peft", "generator": "hf_model"},
        # {"model_name": "openai/gpt-4-1106-preview", "adapter_name": None, "generator_type": None, "generator": "openai"},
        # {"model_name": "openai/gpt-3.5-turbo-1106", "adapter_name": None, "generator_type": None, "generator": "openai"},
        # {"model_name": "replicate/meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3", "adapter_name": None, "generator_type": None, "generator": "replicate"},
        {"model_name": "meta-llama/Llama-2-70b-chat-hf", "adapter_name": None, "generator_type": None, "generator": "hf_inference_api"},
        # {"model_name": "tiiuae/falcon-40b-instruct", "adapter_name": None, "generator_type": None, "generator": "hf_inference_api"},
    ]


    
    for m in models:
        print(f"Generate answer for model: {m['model_name']}, adapter: {m['adapter_name']}, generator type: {m['generator_type']}, generator: {m['generator']}")

        if "openai" == m["generator"]:
            print(f"Generator: {m['generator']}")
            oamodel = OpenAIModel(model_name=m["model_name"].split("/")[1], configs=configs)
            
            for l in d:
                print(f"Test input: {l['test_input']}, Test task: {l['test_task']}, Test context: {l['test_context']}")

                output = oamodel.generate(l["test_input"], l["test_task"], l["test_context"])
                print(f"Output: {output}")
                if IS_ABLATION_LENGTH_TEST:
                    print(f"Author response length: {len(l['test_response'])} | Output response length: {len(output)}")
                    print("--Test with response length--")
                    abt_output = oamodel.generate(l["test_input"], l["test_task"], l["test_context"], response_length=len(l["test_response"]))
                    print(f"Ablation length test output: {abt_output}")
                    print(f"Author response length: {len(l['test_response'])} | Output response length: {len(abt_output)}")
                    print(f"Response length without ablation: {len(output)} | Response length with ablation: {len(abt_output)}")
                print("\n\n\n")

        elif "replicate" == m["generator"]:
            print(f"Generator: {m['generator']}")

            model_name = "/".join(m["model_name"].split("/")[1:])
            print(f"Model name: {model_name}")
            replicate_model = ReplicateModel(model_name=model_name, configs=configs)
            
            for l in d:
                print(f"Test input: {l['test_input']}, Test task: {l['test_task']}, Test context: {l['test_context']}")
 
                output = replicate_model.generate(l["test_input"], l["test_task"], l["test_context"])
                print(f"Output: {output}")
                if IS_ABLATION_LENGTH_TEST:
                    print(f"Author response length: {len(l['test_response'])} | Output response length: {len(output)}")
                    print("--Test with response length--")
                    abt_output = replicate_model.generate(l["test_input"], l["test_task"], l["test_context"], response_length=len(l["test_response"]))
                    print(f"Ablation length test output: {abt_output}")
                    print(f"Author response length: {len(l['test_response'])} | Output response length: {len(abt_output)}")
                    print(f"Response length without ablation: {len(output)} | Response length with ablation: {len(abt_output)}")
                print("\n\n\n")

        elif "hf_inference_api" == m["generator"]:
            print(f"Generator: {m['generator']}")
            hf_inference_model = HFInferenceModel(model_name=m["model_name"], configs=configs)
            
            for l in d:
                print(f"Test input: {l['test_input']}, Test task: {l['test_task']}, Test context: {l['test_context']}")
            
                output = hf_inference_model.generate(l["test_input"], l["test_task"], l["test_context"])
                print(f"Output: {output}")
                if IS_ABLATION_LENGTH_TEST:
                    print(f"Author response length: {len(l['test_response'])} | Output response length: {len(output)}")
                    print("--Test with response length--")
                    abt_output = hf_inference_model.generate(l["test_input"], l["test_task"], l["test_context"], response_length=len(l["test_response"]))
                    print(f"Ablation length test output: {abt_output}")
                    print(f"Author response length: {len(l['test_response'])} | Output response length: {len(abt_output)}")
                    print(f"Response length without ablation: {len(output)} | Response length with ablation: {len(abt_output)}")
                print("\n\n\n")
        
        elif "hf_model" == m["generator"]:
            print(f"Generator: {m['generator']}")
            hfmodel = HFModel(m["model_name"], adapter_name=m["adapter_name"], generator_type=m["generator_type"], configs=configs)
            hfmodel.load_model()
            
            for l in d:
                print(f"Test input: {l['test_input']}, Test task: {l['test_task']}, Test context: {l['test_context']}")

                output = hfmodel.generate(l["test_input"], l["test_task"], l["test_context"])
                print(f"Output: {output}")
                if IS_ABLATION_LENGTH_TEST:
                    print(f"Author response length: {len(l['test_response'])} | Output response length: {len(output)}")
                    print("--Test with response length--")
                    abt_output = hfmodel.generate(l["test_input"], l["test_task"], l["test_context"], response_length=len(l["test_response"]))
                    print(f"Ablation length test output: {abt_output}")
                    print(f"Author response length: {len(l['test_response'])} | Output response length: {len(abt_output)}")
                    print(f"Response length without ablation: {len(output)} | Response length with ablation: {len(abt_output)}")
                print("\n\n\n")

        else:
            raise ValueError("Generator not supported")
            
            
