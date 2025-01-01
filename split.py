from sklearn.model_selection import train_test_split
import pandas as pd
# Load the combined dataset
df = pd.read_csv("combined_abstracts.csv")

# Split into 80% training and 20% testing
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['abstract'], df['label'], test_size=0.2, random_state=42
)

print(f"Training samples: {len(train_texts)}, Testing samples: {len(test_texts)}")
