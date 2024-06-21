import pandas as pd
from loguru import logger
import argparse
import time
import requests
from decouple import config
import datetime
import json

logger.add("../logs/detect_with_gpt_zero.log", rotation="1 MB")


class GPTZeroAPI:
    def __init__(self, api_key, in_file_name=""):
        self.api_key = api_key
        self.base_url = 'https://api.gptzero.me/v2/predict'
        self.retry_count = 3
        self.try_count = 0
        self.request_count = 0
        self.response_list = []
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.checkpoint_step = 2000
        self.in_file_name = in_file_name

    def _text_predict(self, document):
        url = f'{self.base_url}/text'
        headers = {
            'Accept': 'application/json',
            'x-api-key': self.api_key,
            'Content-Type': 'application/json'
        }
        data = {
            'document': document
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()
    

    def get_gpt_zero_detector(self, text, element_id):
        # Check if text is a string
        if not isinstance(text, str):
            logger.error(f"Text is not a string: {text} || Returning None")
            return None, None, None, None, None, None, None, None, None, None
        try:
            response = self._text_predict(text)
            # print(response)
            # return response['predictions'][0]['probability']

            # Backup responses
            self.request_count += 1
            self.response_list.append({
                    'element_id': element_id,
                    'response': response,
                    'text': text
                })
            if self.request_count % self.checkpoint_step == 0:
                logger.info(f"Request count: {self.request_count}")
                logger.info(f"Saving checkpoint...")
                with open(f"../results/gpt_zero_checkpoints/{self.timestamp}_{self.in_file_name}_gpt_zero_responses_{self.request_count}.ckpt.json", "w") as f:
                    json.dump(self.response_list, f)

            return response['documents'][0]['average_generated_prob'], \
                    response['documents'][0]['completely_generated_prob'], \
                    response['documents'][0]['confidence_score'], \
                    response['documents'][0]['overall_burstiness'], \
                    response['documents'][0]['result_message'], \
                    response['documents'][0]['document_classification'], \
                    response['documents'][0]['class_probabilities']['ai'], \
                    response['documents'][0]['class_probabilities']['human'], \
                    response['documents'][0]['class_probabilities']['mixed'],\
                    response['documents'][0]['version']
        
        except Exception as e:
            logger.error(f"Error in gpt_zero_detector: {e}")

            if self.try_count < self.retry_count:
                self.try_count += 1
                logger.info(f"Retrying {self.try_count}/{self.retry_count}")
                # Wait 5 seconds
                time.sleep(5)
                return self.get_gpt_zero_detector(text, element_id)
            
            self.try_count = 0
            return None, None, None, None, None, None, None, None, None, None

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
    logger.info(f"Columns: {df.columns}")

    # time detection
    logger.info("Initiating GPTZeroAPI")
    gpt_zero_api = GPTZeroAPI(config('GPT_ZERO_API_KEY'), in_file_name=args.in_file)

    logger.info("Detecting ...")
    start_time = time.time()
    # Responses;
    # response['documents'][0]['average_generated_prob'], \
    # response['documents'][0]['completely_generated_prob'], \
    # response['documents'][0]['confidence_score'], \
    # response['documents'][0]['overall_burstiness'], \
    # response['documents'][0]['result_message'], \
    # response['documents'][0]['document_classification'], \
    # response['documents'][0]['class_probabilities']['ai'], \
    # response['documents'][0]['class_probabilities']['human'], \
    # response['documents'][0]['class_probabilities']['mixed'],\
    # response['documents'][0]['version']
    df['detector_gpt_zero_average_generated_prob'], \
    df['detector_gpt_zero_completely_generated_prob'], \
    df['detector_gpt_zero_confidence_score'], \
    df['detector_gpt_zero_overall_burstiness'], \
    df['detector_gpt_zero_result_message'], \
    df['detector_gpt_zero_document_classification'], \
    df['detector_gpt_zero_class_probabilities_ai'], \
    df['detector_gpt_zero_class_probabilities_human'], \
    df['detector_gpt_zero_class_probabilities_mixed'],\
    df['detector_gpt_zero_api_version'] = zip(*df.apply(lambda row: gpt_zero_api.get_gpt_zero_detector(row['response'], row['id']), axis=1))
    
    logger.info(f"Time taken: {(time.time() - start_time):.2f} seconds")

    # Sanity check
    logger.info("Sanity check")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Head: {df.head()}")
    logger.info(f"Columns: {df.columns}")

    logger.info("Saving to csv")
    df.to_csv(f'../results/{args.in_file}_gptzero.csv', index=False, escapechar='\\')
    df.to_csv(f'../results/{args.in_file}_gptzero.tsv', index=False, sep="\t", escapechar='\\')