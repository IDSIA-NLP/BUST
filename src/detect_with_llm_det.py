import pandas as pd
from loguru import logger
import argparse
import llmdet
import uuid

# !  -----------------  
# !  The LLMDet model needs the conda environment `llmdet` to be activated
# !  -----------------

llmdet.load_probability()

logger.add("../logs/detect_with_llm_det.log", rotation="1 MB")


def get_llm_det(text):
    try:
        result = llmdet.detect(text)
        print(result)

        return None, None
    except Exception as e:
        logger.error(f"Error in get_xlmr_chatgptdetect_noisy: {e}")
        return None, None

if __name__ == "__main__":
    logger.info("Start")
    # Load data
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="[EXISTING MELTED STATS DATASET FILE]")
    
    args = parser.parse_args()

    in_file_path = f"/FILE/PATH/{args.in_file}.csv"

    logger.info("Starting ...")
    df = pd.read_csv(in_file_path)
    logger.info(f"Read file {in_file_path}")
    logger.info(f"Shape: {df.shape}")

    # Add tmp_id
    df["tmp_id"] = [uuid.uuid4().hex for _ in range(len(df.index))]

    # Non string values lead to an error in llmdet prediction
    tmp_dict = df.dropna(subset=['response'])[['tmp_id', 'response']].to_dict(orient='list')

    # Results are a list of dicts
    logger.info("Getting LLMDet predictions")
    results = llmdet.detect(tmp_dict['response'])
    # {'GPT-2': 0.4773059761375368, 'Human_write': 0.16928650277107687, 'T5': 0.1312581286220548, 'OPT': 0.12779966327850656, 'UniLM': 0.07586656923929233, 'LLaMA': 0.015073496811844396, 'GPT-neo': 0.0018380481043335473, 'Bloom': 0.0010952989113425422, 'BART': 0.0004763161240122074}
    for model in ['GPT-2', 'Human_write', 'T5', 'OPT', 'UniLM', 'LLaMA', 'GPT-neo', 'Bloom', 'BART']:
        tmp_dict[f'detector_llm_det_{model}'] = [x[model] for x in results]
    
    # Add llmdet to df
    df_tmp = pd.DataFrame.from_dict(tmp_dict)
    # Drop response
    df_tmp.drop(columns=['response'], inplace=True)
    # Merge with original df
    df = df.merge(df_tmp, on='tmp_id', how='left')
    # Drop tmp_id
    df.drop(columns=['tmp_id'], inplace=True)


    # Sanity check
    logger.info("Sanity check")
    logger.info(f"Shape: {df.shape}")
    logger.info(df.head())
    logger.info(df.columns)

    logger.info("Saving to csv")
    df.to_csv(f'../results/{args.in_file}_llmdet.csv', index=False, escapechar='\\')
    df.to_csv(f'../results/{args.in_file}_llmdet.tsv', index=False, sep="\t", escapechar='\\')

