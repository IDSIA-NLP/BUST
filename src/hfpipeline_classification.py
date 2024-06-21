# Use a pipeline as a high-level helper
from typing import Any
from transformers import pipeline
import os
import pandas as pd
from argparse import ArgumentParser
from os import environ, path
import torch
from tqdm import tqdm
import logging

env = os.getenv

logger = logging.getLogger(__name__)
logger.setLevel(logging.getLevelName(env("LOGLEVEL", "INFO")))

AVAIL_GPUS =  min(1, torch.cuda.device_count())

DATA_PATH = ""
pipe = None

# Load model directly
#from transformers import AutoTokenizer, AutoModelForSequenceClassification
#tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-irony")
#model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-irony")

class HFPipeline():

    def data(self, data_path, max_len, t10sec, sep="\t"):
        #yield dataset for pipeline
        df = pd.read_csv(data_path, sep=sep)
        if t10sec:
            df = df.iloc[:10,:]

        for idx, e in df.iterrows():
            if t10sec:
                print(f"{idx}--> {str(e['response'])[:max_len]}")
            elif idx%10 == 0 :
                print(idx) # for tracking progress
            yield str(e['response'])[:max_len]

    def __call__(self, data_path, model, max_len, tgt_path, t10sec=False, **kwargs) -> str:

        logger.info(f"Available GPUs: {AVAIL_GPUS}, model: {model}, data: {data_path} --> {tgt_path}")
        pipe = pipeline("text-classification", model=model, return_all_scores=True, function_to_apply="softmax")

        rs = []
        for out in pipe(self.data(data_path, max_len, t10sec)): #, batch_size=4): # no gain when batching
            rs.append(out)

        rs = pd.DataFrame(rs)
        rs.to_csv(tgt_path, sep="\t")

        # returns the path of the created file
        return tgt_path 

# =========================================

def main(**kwargs):
    classify = HFPipeline()
    classify(**kwargs)

if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve') # add_help=False to be used in cluster

    parser.add_argument('--t10sec', type=bool, default=False, help="Sanity check (Unitest)")
    parser.add_argument('--data_path', type=str, default="./data/20230808_092322_databricks-dolly_size-200_source.tsv", help="Dataset (as tsv)")
    # parser.add_argument('--model', type=str, default="kinit/semeval2023-task3-persuasion-techniques")
    # parser.add_argument('--model', type=str, default="SamLowe/roberta-base-go_emotions")
    # parser.add_argument('--model', type=str, default="cardiffnlp/twitter-roberta-base-irony")
    parser.add_argument('--model', type=str, default="jakub014/bert-base-uncased-IBM-argQ-30k-finetuned-convincingness-IBM")
    parser.add_argument('--max_len', type=int, default=350, help="Max se length passed to the pipeline (if longer is truncated)")
    parser.add_argument('--tgt_path', type=str, default="convincing_softmax.tsv", help="path_name to save the output.")

    # args = parser.parse_args()
    args, other_args = parser.parse_known_args()

    main(**vars(args))

