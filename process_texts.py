# preprocess_texts.py
import pandas as pd
from transformers import CLIPTokenizer

JSON_PATH = './dataset/arrow_dataset/annotations.json'
MODEL_ID = "runwayml/stable-diffusion-v1-5"

df = pd.read_json(JSON_PATH, encoding='utf-8')
tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
df['text_input_ids'] = df['coaching_text'].apply(lambda t: tokenizer(t, padding='max_length', max_length=tokenizer.model_max_length, truncation=True).input_ids)
df.to_json('dataset/arrow_dataset/annotations_with_tokens.json', orient='records', force_ascii=False)