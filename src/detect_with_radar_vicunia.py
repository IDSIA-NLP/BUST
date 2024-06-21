import transformers
import torch
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
from loguru import logger
import argparse
import time

# Based on the code from https://colab.research.google.com/drive/1r7mLEfVynChUUgIfw1r4WZyh9b0QBQdo?usp=sharing#scrollTo=OlVWEymLRhTl 

logger.add("../logs/detect_with_radar_vicunia.log", rotation="1 MB")

logger.info("Preparing models")
# device = "cpu"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

model_name ="TrustSafeAI/RADAR-Vicuna-7B"
logger.info(f"Loading model {model_name}")
logger.info("This might take a while ...")
detector = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
detector.eval()
detector.to(device)


def get_radar_vicuna_7B_detector(text):
    try:
        with torch.no_grad():
            inputs = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k:v.to(device) for k,v in inputs.items()}
            output_probs = F.log_softmax(detector(**inputs).logits,-1)[:,0].exp().tolist()
            # logger.debug(f"There are {len([text])} input instances")
            # logger.debug(f"Probability of AI-generated texts is {output_probs}")
            
        return output_probs[0]
    except Exception as e:
        logger.error(f"Error in radar_vicuna_7B_detector: {e}")
        return None

if __name__ == "__main__":
    logger.info("------ Start ------")
    # Load data
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="[EXISTING MELTED STATS DATASET FILE]")
    
    args = parser.parse_args()

    in_file_path = f"/FILE/PATH/{args.in_file}.csv"

    logger.info("Starting ...")
    df = pd.read_csv(in_file_path)
    logger.info(f"Read file {in_file_path}")
    logger.info(f"Shape: {df.shape}")
   
    # time detection
    logger.info("Detecting ...")
    start_time = time.time()
    df['detector_radar_vicuna_7B_ai_prob'] = df['response'].map(get_radar_vicuna_7B_detector)
    
    logger.info(f"Time taken: {(time.time() - start_time):.2f} seconds")

    # Sanity check
    logger.info("Sanity check")
    logger.info(f"Shape: {df.shape}")
    logger.info(df.head())
    logger.info(df.columns)

    logger.info("Saving to csv")
    df.to_csv(f'../results/{args.in_file}_radar.csv', index=False, escapechar='\\')
    df.to_csv(f'../results/{args.in_file}_radar.tsv', index=False, sep="\t", escapechar='\\')

