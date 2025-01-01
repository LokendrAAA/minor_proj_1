import pandas as pd
import torch
from transformers import RobertaTokenizer

def load_data():
    """Load the combined dataset."""
    df = pd.read_csv("combined_abstracts.csv")
    df = df.dropna(subset=['abstract'])  # Clean missing abstracts
    df['abstract'] = df['abstract'].astype(str)  # Ensure all abstracts are strings
    return df['abstract'], df['label']

def preprocess_data(texts, labels):
    """Tokenize texts and convert labels to tensors."""
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    if isinstance(texts, pd.Series):
        texts = texts.dropna().astype(str).tolist()

    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    labels = torch.tensor(labels.values)
    return encodings, labels
