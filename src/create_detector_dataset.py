# Use the detectors to predict for eacht continuatioon of the dataset if its human or generated

import pandas as pd
from readability import Readability
import nltk
import string
import spacy

from gradio_client import Client
import json

from loguru import logger
logger.add("../logs/create_detector_dataset.log", rotation="1 MB")

in_file_name = "[EXISTING MELTED DATASET FILE]"


df = pd.read_csv(f'../results/final_datasets/{in_file_name}.csv')

logger.info("Starting create_detector_dataset.py")

logger.info(f"Loaded {in_file_name}.csv")
logger.info(f"Shape: {df.shape}")
logger.debug(f"Head: {df.head()}")
# print(df.head())



metrics = [
    # Readability
    'readability_flesch_kincaid', 
    'readability_flesch',
    'readability_gunning_fog',
    'readability_coleman_liau',
    'readability_dale_chall',
    'readability_ari',
    'readability_linsear_write',
    'readability_spache'
    # Statistical measurments
    'text_length',
    'avg_word_length',
    'avg_sentence_length',
    'avg_words_per_sentence',
    'percent_vowels',
    'percent_consonants',
    'percent_punctuation',
    'percent_stopwords',
    'num_words',
    'num_sentences',
    'percent_unique_words',
    'percent_long_words',
    # Spacy
    'percent_nouns',
    'percent_verbs',
    'percent_adjectives',
    # Detectors
    'chatgpt_qa_detector_roberta',
    'xlmr_chatgptdetect_noisy',
    # 'radar_vicuna_7B_detector',
]

# --------------------------------------------------------------------------------------------
#                                      Readability metrics
# --------------------------------------------------------------------------------------------


def get_readability_flesch_kincaid(text):
    try:
        r = Readability(text)
        fk = r.flesch_kincaid()
        return fk.score
    except Exception as e:
        # logger.error(e)
        return None


def get_readability_flesch(text):
    try:
        r = Readability(text)
        f = r.flesch()
        return f.score
    except Exception as e:
        # logger.error(e)
        return None
    

def get_readability_gunning_fog(text):  
    try:
        r = Readability(text)
        gf = r.gunning_fog()
        return gf.score
    except Exception as e:
        # logger.error(e)
        return None
    

def get_readability_coleman_liau(text):
    try:
        r = Readability(text)
        cl = r.coleman_liau()
        return cl.score
    except Exception as e:
        # logger.error(e)
        return None
    

def get_readability_dale_chall(text):
    try:
        r = Readability(text)
        dc = r.dale_chall()
        return dc.score
    except Exception as e:
        # logger.error(e)
        return None
    

def get_readability_ari(text):
    try:
        r = Readability(text)
        ari = r.ari()
        return ari.score
    except Exception as e:
        # logger.error(e)
        return None
    

def get_readability_linsear_write(text):
    try:
        r = Readability(text)
        lw = r.linsear_write()
        return lw.score
    except Exception as e:
        # logger.error(e)
        return None
    

def get_readability_spache(text):
    try:
        r = Readability(text)
        s = r.spache()
        return s.score, s.grade_levels
    except Exception as e:
        # logger.error(e)
        return None
    



if "readability_flesch_kincaid" in metrics:
    logger.info("Getting readability_flesch_kincaid")
    df['readability_flesch_kincaid_score'] = df['response'].map(get_readability_flesch_kincaid)

if "readability_flesch" in metrics:
    logger.info("Getting readability_flesch")
    df['readability_flesch_score']= df['response'].map(get_readability_flesch)

if "readability_gunning_fog" in metrics:
    logger.info("Getting readability_gunning_fog")
    df['readability_gunning_fog_score'] = df['response'].map(get_readability_gunning_fog)

if "readability_coleman_liau" in metrics:
    logger.info("Getting readability_coleman_liau")
    df['readability_coleman_liau_score'] = df['response'].map(get_readability_coleman_liau)

if "readability_dale_chall" in metrics:
    logger.info("Getting readability_dale_chall") 
    df['readability_dale_chall_score'] = df['response'].map(get_readability_dale_chall)

if "readability_ari" in metrics:
    logger.info("Getting readability_ari")
    df['readability_ari_score'] = df['response'].map(get_readability_ari)

if "readability_linsear_write" in metrics:
    logger.info("Getting readability_linsear_write")
    df['readability_linsear_write_score'] = df['response'].map(get_readability_linsear_write)

if "readability_spache" in metrics:
    logger.info("Getting readability_spache")
    df['readability_spache_score'] = df['response'].map(get_readability_spache)

logger.info("Done readability")
logger.debug(f"Head: {df.head()}")

# get mean of each metric for each model

df_mean = df.groupby(['model'])[[col for col in df.columns if col.endswith('_score')]].mean()
logger.debug(f"Head df_mean: {df_mean.head()}")


# --------------------------------------------------------------------------------------------
#                                      Statistical metrics
# --------------------------------------------------------------------------------------------

def get_text_length(text):
    try:
        return len(text)
    except Exception as e:
        logger.error(f"Error in get_text_length: {e}")
        return None


def get_avg_word_length(text):
    try:
        words = text.split(' ')
        return sum([len(word) for word in words]) / len(words)
    except Exception as e:  
        logger.error(f"Error in get_avg_word_length: {e}")
        return None
    

def get_avg_sentence_length(text):
    try:
        sentences = text.split('.')
        return sum([len(sentence) for sentence in sentences]) / len(sentences)
    except Exception as e:
        logger.error(f"Error in get_avg_sentence_length: {e}")
        return None
    

def get_avg_words_per_sentence(text):
    try:
        sentences = text.split('.')
        return sum([len(sentence.split(' ')) for sentence in sentences]) / len(sentences)
    except Exception as e:
        logger.error(f"Error in get_avg_words_per_sentence: {e}")
        return None
    
def get_percent_vowels(text):
    try:
        vowels = ['a', 'e', 'i', 'o', 'u']
        return sum([1 for char in text if char in vowels]) / len(text)
    except Exception as e:
        logger.error(f"Error in get_percent_vowels: {e}")
        return None
    

def get_percent_consonants(text):
    try:
        consonants = [char for char in string.ascii_lowercase if char not in ['a', 'e', 'i', 'o', 'u']]
        return sum([1 for char in text if char in consonants]) / len(text)
    except Exception as e:
        logger.error(f"Error in get_percent_consonants: {e}")
        return None
    

def get_percent_punctuation(text):
    try:
        punctuation = string.punctuation
        return sum([1 for char in text if char in punctuation]) / len(text)
    except Exception as e:
        logger.error(f"Error in get_percent_punctuation: {e}")
        return None
    
def get_percent_stopwords(text):
    try:
        stopwords = nltk.corpus.stopwords.words('english')
        return sum([1 for word in text.split(' ') if word in stopwords]) / len(text.split(' '))
    except Exception as e:
        logger.error(f"Error in get_percent_stopwords: {e}")
        return None
    

def get_num_words(text):
    try:
        return len(text.split(' '))
    except Exception as e:
        logger.error(f"Error in get_num_words: {e}")
        return None

def get_num_sentences(text):
    try:
        return len(text.split('.'))
    except Exception as e:
        logger.error(f"Error in get_num_sentences: {e}")
        return None

def get_percent_unique_words(text):
    try:
        return len(set(text.split(' '))) / len(text.split(' '))
    except Exception as e:
        logger.error(f"Error in get_percent_unique_words: {e}")
        return None

def get_percent_long_words(text):
    try:
        return len([word for word in text.split(' ') if len(word) > 7]) / len(text.split(' '))
    except Exception as e:
        logger.error(f"Error in get_percent_long_words: {e}")
        return None



if "text_length" in metrics:
    logger.info("Getting text_length")
    df['stats_text_length'] = df['response'].map(get_text_length)

if "avg_word_length" in metrics:
    logger.info("Getting avg_word_length")
    df['stats_avg_word_length'] = df['response'].map(get_avg_word_length)

if "avg_sentence_length" in metrics:
    logger.info("Getting avg_sentence_length")
    df['stats_avg_sentence_length'] = df['response'].map(get_avg_sentence_length)

if "avg_words_per_sentence" in metrics:
    logger.info("Getting avg_words_per_sentence")
    df['stats_avg_words_per_sentence'] = df['response'].map(get_avg_words_per_sentence)

if "percent_vowels" in metrics:
    logger.info("Getting percent_vowels")
    df['stats_percent_vowels'] = df['response'].map(get_percent_vowels)

if "percent_consonants" in metrics:
    logger.info("Getting percent_consonants")
    df['stats_percent_consonants'] = df['response'].map(get_percent_consonants)

if "percent_punctuation" in metrics:
    logger.info("Getting percent_punctuation")
    df['stats_percent_punctuation'] = df['response'].map(get_percent_punctuation)

if "percent_stopwords" in metrics:
    logger.info("Getting percent_stopwords")
    df['stats_percent_stopwords'] = df['response'].map(get_percent_stopwords)

if "num_words" in metrics:
    logger.info("Getting num_words")
    df['stats_num_words'] = df['response'].map(get_num_words)

if "num_sentences" in metrics:
    logger.info("Getting num_sentences")
    df['stats_num_sentences'] = df['response'].map(get_num_sentences)

if "percent_unique_words" in metrics:
    logger.info("Getting percent_unique_words")
    df['stats_percent_unique_words'] = df['response'].map(get_percent_unique_words)

if "percent_long_words" in metrics:
    logger.info("Getting percent_long_words")
    df['stats_percent_long_words'] = df['response'].map(get_percent_long_words)

logger.info("Done stats")
logger.debug(f"Head: {df.head()}")

# --------------------------------------------------------------------------------------------
#                                      spacy-based metrics
# --------------------------------------------------------------------------------------------

nlp = spacy.load('en_core_web_sm')

def get_percent_nouns(text):
    try:
        doc = nlp(text)
        return len([token for token in doc if token.pos_ == 'NOUN']) / len([token for token in doc])
    except Exception as e:
        logger.error(f"Error in get_num_nouns: {e}")
        return None

def get_percent_verbs(text):
    try:
        doc = nlp(text)
        return len([token for token in doc if token.pos_ == 'VERB']) / len([token for token in doc])
    except Exception as e:
        logger.error(f"Error in get_num_verbs: {e}")
        return None


def get_percent_adjectives(text):
    try:
        doc = nlp(text)
        return len([token for token in doc if token.pos_ == 'ADJ']) / len([token for token in doc])
    except Exception as e:
        logger.error(f"Error in get_num_adjectives: {e}")
        return None

def get_sentiment(text):
    try:
        doc = nlp(text)
        return doc.sentiment
    except Exception as e:
        logger.error(f"Error in get_sentiment: {e}")
        return None

if "percent_nouns" in metrics:
    logger.info("Getting percent_nouns")
    df['spacy_percent_nouns'] = df['response'].map(get_percent_nouns)

if "percent_verbs" in metrics:
    logger.info("Getting percent_verbs")
    df['spacy_percent_verbs'] = df['response'].map(get_percent_verbs)

if "percent_adjectives" in metrics:
    logger.info("Getting percent_adjectives")
    df['spacy_percent_adjectives'] = df['response'].map(get_percent_adjectives)

if "sentimenta" in metrics:
    logger.info("Getting sentiment")
    df['spacy_sentiment'] = df['response'].map(get_sentiment)

logger.info("Done spacy")
logger.debug(f"Head: {df.head()}")


# --------------------------------------------------------------------------------------------
#                                      Detectors
# --------------------------------------------------------------------------------------------
from transformers import pipeline
from transformers import pipeline

try:
    logger.info("Loading detector pipelines")
    if "chatgpt_qa_detector_roberta" in metrics:
        logger.info("Loading chatgpt_qa_detector_roberta")
        pipe_chatgpt_qa = pipeline("text-classification", model="Hello-SimpleAI/chatgpt-qa-detector-roberta",  max_length=512, truncation=True)
    if "xlmr_chatgptdetect_noisy" in metrics:
        logger.info("Loading xlmr_chatgptdetect_noisy")
        pipe_xlmr = pipeline("text-classification", model="almanach/xlmr-chatgptdetect-noisy",  max_length=512, truncation=True)
    if "radar_vicuna_7B_detector" in metrics:
        logger.info("Loading radar_vicuna_7B_detector")
        radar_vicuna_7B_gradio_client = Client("https://trustsafeai-radar-ai-text-detector.hf.space/")
    
except Exception as e:  
    logger.error(f"Error in loading detector pipelines {e}")


def get_chatgpt_qa_detector_roberta(text):
    try:
        result = pipe_chatgpt_qa(text)
        # logger.info(result)
        return result[0]['score'], result[0]['label']
    except Exception as e:
        logger.error(f"Error in get_chatgpt_qa_detector_roberta: {e}")
        return None, None


def get_xlmr_chatgptdetect_noisy(text):
    try:
        result = pipe_xlmr(text)
        # print(result)
        return result[0]['score'], result[0]['label']
    except Exception as e:
        logger.error(f"Error in get_xlmr_chatgptdetect_noisy: {e}")
        return None, None
    

def read_radar_json(file_path):
    try:
        with open(file_path, 'r') as f:
            radar_results = json.load(f)
        return radar_results
    except Exception as e:
        logger.error(f"Error in read_radar_json: {e}")
        return None

def get_radar_vicuna_7B_detector(text):
    def extract_confidences(confidences):
        human_confidence = next((item['confidence'] for item in confidences if item['label'] == 'Human-written'), 0)
        ai_confidence = next((item['confidence'] for item in confidences if item['label'] == 'AI-generated'), 0)
        return human_confidence, ai_confidence
    try:
        
        result_file_path = radar_vicuna_7B_gradio_client.predict(
            text,	# str  in 'Text to be detected' Textbox component
            api_name="/predict"
        )
        result_json = read_radar_json(result_file_path)
        human_conf_score, ai_conf_score = extract_confidences(result_json['confidences'])
        # print(result_json['label'], human_conf_score, ai_conf_score)
        return result_json['label'], human_conf_score, ai_conf_score
    
    except Exception as e:
        logger.error(f"Error in get_radar_vicuna_7B_detector: {e}")
        return None, None, None

if "chatgpt_qa_detector_roberta" in metrics:
    logger.info("Getting chatgpt_qa_detector_roberta")
    df['detector_chatgpt_qa_detector_score'], df['detector_chatgpt_qa_detector_label'] = zip(*df['response'].map(get_chatgpt_qa_detector_roberta))

if "xlmr_chatgptdetect_noisy" in metrics:
    logger.info("Getting xlmr_chatgptdetect_noisy")
    df['detector_xlmr_chatgptdetect_noisy_score'], df['detector_xlmr_chatgptdetect_noisy_label'] = zip(*df['response'].map(get_xlmr_chatgptdetect_noisy))

if "radar_vicuna_7B_detector" in metrics:
    logger.info("Getting radar_vicuna_7B_detector")
    df['detector_radar_vicuna_7B_label'], df['detector_radar_vicuna_7B_human_confidence_score'], df['detector_radar_vicuna_7B_ai_confidence_score'] = zip(*df['response'].map(get_radar_vicuna_7B_detector)) 


logger.info("Done detectors")
logger.info(f"Head: {df.head()}")
# --------------------------------------------------------------------------------------------
#                                     Save to csv   
# --------------------------------------------------------------------------------------------

logger.info("Saving data")
df.to_csv(f'../results/{in_file_name}_stats.csv', index=False, escapechar='\\')
df.to_csv(f'../results/{in_file_name}_stats.tsv', index=False, sep="\t", escapechar='\\')