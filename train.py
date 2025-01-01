import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaForSequenceClassification, RobertaTokenizer, AdamW
from sklearn.model_selection import train_test_split
from utils import load_data, preprocess_data
from tqdm import tqdm

def train_model():
    texts, labels = load_data()
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    train_encodings, train_labels = preprocess_data(train_texts, train_labels)

    train_dataset = TensorDataset(
        train_encodings['input_ids'], train_encodings['attention_mask'], train_labels
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(3):
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in epoch_iterator:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_iterator.set_postfix(loss=loss.item())
        print(f"Epoch {epoch + 1} completed")
    
    # Save the model and tokenizer
    model.save_pretrained("path_to_your_model_directory1")  # Save model and config
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.save_pretrained("path_to_your_model_directory1")  # Save tokenizer


if __name__ == "__main__":
    train_model()
