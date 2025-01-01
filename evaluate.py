from torch.utils.data import DataLoader, TensorDataset
import torch
from utils import load_data, preprocess_data
from sklearn.model_selection import train_test_split
from transformers import RobertaForSequenceClassification

def evaluate_model(model, val_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    print(f"Number of batches in val_loader: {len(val_loader)}")  # Check number of batches

    with torch.no_grad():  # Disable gradient calculation
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            
            # Check the contents of the batch
            print(f"Input IDs: {input_ids}, Attention Mask: {attention_mask}, Labels: {labels}")

            outputs = model(input_ids, attention_mask=attention_mask)
            print(f"Outputs: {outputs.logits}")  # Print the outputs for debugging
            
            predictions = torch.argmax(outputs.logits, dim=1)  # Get predicted labels

            correct += (predictions == labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Count total samples

    if total > 0:  # Avoid division by zero
        accuracy = correct / total * 100  # Calculate accuracy
        print(f"Validation Accuracy: {accuracy:.2f}%")
    else:
        print("No samples to evaluate.")

if __name__ == "__main__":
    # Load data
    texts, labels = load_data()  # Load the complete dataset
    
    # Correctly unpack train_test_split into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    val_encodings, val_labels = preprocess_data(val_texts, val_labels)  # Preprocess the validation data

    # Create dataset using tensor directly from labels
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val_labels.tolist()))  # Convert val_labels to list
    val_loader = DataLoader(val_dataset, batch_size=8)  # Create DataLoader

    # Load your trained model
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.load_state_dict(torch.load("path_to_your_model.pth"))  # Load your trained model state

    evaluate_model(model, val_loader)
