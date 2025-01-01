import pandas as pd

# Load human-written abstracts (assign label 1)
human_df = pd.read_csv("abstracts.csv")
human_df['label'] = 1  # Human-written = 1

# Load AI-generated abstracts (assign label 0)
ai_df = pd.read_csv("aiabstracts.csv")
ai_df['label'] = 0  # AI-generated = 0

# Combine both datasets
combined_df = pd.concat([human_df, ai_df], ignore_index=True)

# Shuffle the dataset to avoid any order bias
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# Save the combined dataset to CSV (optional)
combined_df.to_csv("combined_abstracts.csv", index=False)
print(combined_df.head())  # Check if everything looks correct
