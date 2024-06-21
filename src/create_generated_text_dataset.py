#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------------------
#                                           IMPORT
# --------------------------------------------------------------------------------------------
import argparse
from loguru import logger
import wandb
import pandas as pd
import datetime
import json
from datasets import load_dataset



from generators import OpenAIModel, HFModel, ReplicateModel, HFInferenceModel

# ---------------------------------------- Setups --------------------------------------
logger.add("../logs/run.log", rotation="1 MB")

wandb.init(project='[PROJECT]',  entity="[ENTITY]", reinit=True)


TASK_NAME = {
    "open_qa": "Open-domain Question Answering",
    "general_qa": "General Question Answering",
    "classification": "Classification",
    "closed_qa": "Closed-domain Question Answering",  
    "brainstorming": "Brainstorming",
    "information_extraction": "Information Extraction",
    "summarization": "Summarization",
    "creative_writing": "Creative Writing",
    }
# --------------------------------------------------------------------------------------------
#                                            MAIN
# --------------------------------------------------------------------------------------------


# ----------------------------------------- Functions ----------------------------------------

def get_column_name(model_name, adapter_name, generator):
    if adapter_name:
        return f"{generator}_{model_name}_{adapter_name}"
    else:
        return f"{generator}_{model_name}"
    

def main(args, models, configs):
    logger.info(f'Start main...')
    # Assert if more than one flag of update, expand, extension_review and extension_stance are set at the same time
    assert not (args.update and args.expand and args.extension_review and args.extension_stance), "Only one of the flags update, expand, extension_review and extension_stance can be set at the same time."

    
    if args.update:
        # Update dataset by adding generation for a new model
        logger.info(f"Update data from: {args.data_in_path}")
        df_data_sample = pd.read_csv(args.data_in_path)

        dataset_size = df_data_sample.shape[0]

    elif args.expand:
        # Expand dataset by adding more samples of the core dataset
        logger.info(f"Expand data from: {args.data_in_path}")
        df_data_sample_old = pd.read_csv(args.data_in_path)

        logger.info(f"Load dataset: databricks/databricks-dolly-15k")
        dataset = load_dataset("databricks/databricks-dolly-15k")
        df_data = dataset['train'].to_pandas()

        extra_sample_size = args.size * 4
        df_data_sample_extra = df_data.sample(extra_sample_size, random_state=42)

        old_sample_counts = df_data_sample_old['category'].value_counts()

        # Remove the old samples from the extra sample
        df_data_sample_extra_last_n = df_data_sample_extra.iloc[len(df_data_sample_old):]

        extra_data = []

        for category in df_data_sample_extra_last_n['category'].unique():
            needed = args.expand_size - old_sample_counts.get(category, 0)
            if needed > 0:
                print(f"Category: {category} | Needed: {needed}")
                # Filter the original df for the category
                category_rows = df_data_sample_extra_last_n[df_data_sample_extra_last_n['category'] == category]
                    
                # Sample the needed number of rows
                extra_data.append(category_rows.iloc[:needed])
                print(f"Category: {category} | Number of new samples: {extra_data[-1].shape[0]}")
                    
        # Append these rows to the subsample
        df_data_sample = pd.concat(extra_data)

        dataset_size = df_data_sample.shape[0] + df_data_sample_old.shape[0]

    elif args.extension_review:
        logger.info(f"Load extension dataset: Review")
        df_data_sample = pd.read_csv("../data/repurposeds_review_generation.tsv", sep="\t")
        df_data_sample = df_data_sample[["instruction", "context", "category", "response"]]

        dataset_size = df_data_sample.shape[0]
        

    elif args.extension_stance:
        logger.info(f"Load extension dataset: Stance")
        df_data_sample = pd.read_csv("../data/repurposeds_stance_generation.tsv", sep="\t")
        df_data_sample = df_data_sample[["instruction", "context", "category", "response"]]

        dataset_size = df_data_sample.shape[0]
        

    else:
        logger.info(f"Load dataset: databricks/databricks-dolly-15k")
        dataset = load_dataset("databricks/databricks-dolly-15k")
        df_data = dataset['train'].to_pandas()
        
        # Sample the dataset
        # df_data_sample = df_data.sample(args.size, random_state=42)

        # Sample the dataset equally for each category
        new_data = []
        for category in df_data['category'].unique():
            category_rows = df_data[df_data['category'] == category].sample(args.size_per_cat, random_state=42)
            new_data.append(category_rows)
            print(f"Category: {category} | Number of samples: {category_rows.shape[0]}")

        df_data_sample = pd.concat(new_data)
        
        target_size = args.size_per_cat * len(df_data['category'].unique())
        print(f"Sampled data size: {df_data_sample.shape[0]} | Target size: {target_size}")
        dataset_size = df_data_sample.shape[0]

    
    # Define the output file name
    if args.extension_review:
        out_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_review_size-{dataset_size}"
    elif args.extension_stance:
        out_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_stance_size-{dataset_size}"
    else:
        out_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_databricks-dolly_size-{dataset_size}"
    
    # Define complete save file name
    if configs['is_ablation_length']:
        save_file_name = f"{args.data_out_path}{out_name}_ABLATION_LENGTH"
    elif configs['is_ablation_temperature']:
        save_file_name = f"{args.data_out_path}{out_name}_ABLATION_TEMPERATURE_MAX"
    else:
        save_file_name = f"{args.data_out_path}{out_name}"
    logger.info(f"Save file name: {save_file_name}")
    
    for m in models:
        logger.info(f"Generate answer for model: {m['model_name']}, adapter: {m['adapter_name']}, generator type: {m['generator_type']}")

        column_name = get_column_name(m["model_name"], m["adapter_name"], m["generator"])
        if "openai" in m["generator"]:
            
            open_ai_model = OpenAIModel(model_name=m["model_name"].split("/")[1], configs=configs)
            if configs['is_ablation_length']:
                df_data_sample[column_name] = df_data_sample.apply(lambda x: open_ai_model.generate(x.instruction, TASK_NAME.get(x.category, ""), x.context, response_length=len(x.response)), axis=1)
            else:
                df_data_sample[column_name] = df_data_sample.apply(lambda x: open_ai_model.generate(x.instruction, TASK_NAME.get(x.category, ""), x.context), axis=1)
        
        elif "replicate" in m["generator"]:
            model_name = "/".join(m["model_name"].split("/")[1:])
            replicate_model = ReplicateModel(model_name=model_name, configs=configs)
            if configs['is_ablation_length']:
                df_data_sample[column_name] = df_data_sample.apply(lambda x: replicate_model.generate(x.instruction, task=TASK_NAME.get(x.category, ""), context=x.context, response_length=len(x.response)), axis=1)
            else:
                df_data_sample[column_name] = df_data_sample.apply(lambda x: replicate_model.generate(x.instruction, task=TASK_NAME.get(x.category, ""), context=x.context), axis=1)
        
        elif "hf_inference_api" == m["generator"]:
            print(f"Generator: {m['generator']}")
            hf_inference_model = HFInferenceModel(model_name=m["model_name"], configs=configs)
            
            if configs['is_ablation_length']:
                df_data_sample[column_name] = df_data_sample.apply(lambda x: hf_inference_model.generate(x.instruction, task=TASK_NAME.get(x.category, ""), context=x.context, response_length=len(x.response)), axis=1)    
            else:
                df_data_sample[column_name] = df_data_sample.apply(lambda x: hf_inference_model.generate(x.instruction, task=TASK_NAME.get(x.category, ""), context=x.context), axis=1)


        elif "hf_model" == m["generator"]:
            hfmodel = HFModel(m["model_name"], adapter_name=m["adapter_name"], generator_type=m["generator_type"], configs=configs)
            hfmodel.load_model()
            if configs['is_ablation_length']:
                df_data_sample[column_name] = df_data_sample.apply(lambda x: hfmodel.generate(x.instruction, TASK_NAME.get(x.category, ""), x.context, response_length=len(x.response)), axis=1)
            else:
                df_data_sample[column_name] = df_data_sample.apply(lambda x: hfmodel.generate(x.instruction, TASK_NAME.get(x.category, ""), x.context), axis=1)

        else:
            raise ValueError("Generator not supported")
        

        logger.info(f"Save checkpoint to: {save_file_name}.ckpt.csv")
        df_data_sample.to_csv(f"{save_file_name}.ckpt.csv", index=False, escapechar='\\')
        logger.info("Write checkpoint to tsv")
        df_data_sample.to_csv(f"{save_file_name}.ckpt.tsv", index=False, sep="\t", escapechar='\\')
    
    
    if args.expand:
        # Concat the new subsample to the original subsample
        df_data_sample = pd.concat([df_data_sample_old, df_data_sample])

    logger.debug(f"Output df_data_sample: {df_data_sample.shape}")
    
    logger.info(f"Saved to: {save_file_name}.csv")
    df_data_sample.to_csv(f"{save_file_name}.csv", index=False, escapechar='\\')
    logger.info("Write to tsv")
    df_data_sample.to_csv(f"{save_file_name}.tsv", index=False, sep="\t", escapechar='\\')


    # Meta data as intended json
    with open(f"{save_file_name}.meta.json", "w") as f:
        json.dump({
            "dataset": "databricks-dolly",
            "size": args.size,
            "size_per_cat": args.size_per_cat,
            "models": models,
            "date": datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
            "update": args.update,
            "expand": args.expand,
            "expand_size": args.expand_size,
            "extension_review": args.extension_review,
            "extension_stance": args.extension_stance,
            "dataset_size": dataset_size,
            "is_ablation_length": configs['is_ablation_length'],
            "is_ablation_temperature": configs['is_ablation_temperature'],
        }, f, indent=4)

    return 


# --------------------------------------------------------------------------------------------
#                                          RUN
# --------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    logger.info(f'Start ...')

    # Load configs
    with open('../configs/configs.yaml') as f:
        configs = yaml.safe_load(f)

    assert not (configs['is_ablation_length'] and configs['is_ablation_temperature']), "Ablation length and temperature cannot be set at the same time."
    if configs['is_ablation_length']:
        logger.info(f"\n******************** ABLATION LENGTH ********************\n")
    if configs['is_ablation_temperature']:
        logger.info(f"\n******************** ABLATION TEMPERATURE ********************\n")

    # Setup argument parser
    description = "Application description"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_out_path', metavar='data_out_path', type=str, required=False, 
                        default='../results/', help='Path to output data')
    parser.add_argument('-s', '--size', metavar='size', type=int, required=False, default=1000,
                        help='Size of the output dataset')
    parser.add_argument('-u', '--update', metavar='update', type=bool, required=False, default=False,
                        help='Update the dataset by adding generation for a new model')
    parser.add_argument('-exr', '--extension_review', metavar='extension_review', type=bool, required=False, default=False,
                        help='Extension dataset: Review')
    parser.add_argument('-exs', '--extension_stance', metavar='extension_stance', type=bool, required=False, default=False,
                        help='Extension dataset: Stance')
    parser.add_argument('-spc', '--size_per_cat', metavar='size_per_cat', type=int, required=False, default=250,
                        help='Number of samples for each category dataset.')
    parser.add_argument('-i', '--data_in_path', metavar='data_in_path', type=str, required=False, 
                        default='../results/20231024_160028_databricks-dolly_size-1000.csv', help='Path for input data to update.')
    parser.add_argument('-e', '--expand', metavar='expand', type=bool, required=False, default=False,
                        help='Expand the dataset by adding more samples of the core dataset.')
    parser.add_argument('-es', '--expand_size', metavar='expand_size', type=int, required=False, default=200,
                        help='Number of samples to add for each category when expanding the dataset.')
    
    
    args = parser.parse_args()
    


    models = [
        # {"model_name": "meta-llama/Llama-2-7b-chat-hf", "adapter_name": None, "generator_type": "pipeline", "generator": "hf_model"},
        # {"model_name": "tiiuae/falcon-7b-instruct", "adapter_name": None, "generator_type": "pipeline", "generator": "hf_model"},
        # {"model_name": "huggyllama/llama-7b", "adapter_name": 'timdettmers/guanaco-7b', "generator_type": "peft", "generator": "hf_model"},
        # ## {"model_name": "replicate/meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3", "adapter_name": None, "generator_type": None, "generator": "replicate"},
        # ## {"model_name": "replicate/joehoover/falcon-40b-instruct:7eb0f4b1ff770ab4f68c3a309dd4984469749b7323a3d47fd2d5e09d58836d3c", "adapter_name": None, "generator_type": None, "generator": "replicate"},
        {"model_name": "meta-llama/Llama-2-70b-chat-hf", "adapter_name": None, "generator_type": None, "generator": "hf_inference_api"},
        # {"model_name": "openai/gpt-4-1106-preview", "adapter_name": None, "generator_type": None, "generator": "openai"},
        # {"model_name": "openai/gpt-3.5-turbo-1106", "adapter_name": None, "generator_type": None, "generator": "openai"},
        # {"model_name": "tiiuae/falcon-7b", "adapter_name": 'jco2/falcon-7b-dolly-peft-stp-10k', "generator_type": "peft", "generator": "hf_model"},
    ]

   # Run main
    main(args, models, configs)
    
    # Wrapping stuff up
    wandb.finish()
    logger.info("Done!")