# prepare_indic_sentiment_telugu.py

import os
import pandas as pd
from datasets import load_dataset

def download_and_filter_indic_sentiment():
    """
    Download and filter the IndicSentiment dataset for Telugu.
    """
    print("Downloading IndicSentiment dataset...")
    dataset = load_dataset("ai4bharat/IndicSentiment", "translation-te", trust_remote_code=True)
    
    print("Filtering for Telugu data...")
    telugu_data = []
    
    for split in dataset:
        df = dataset[split].to_pandas()
        print(f"Processing {split} split with {len(df)} examples")
        
        # Keep only necessary columns and rename them
        telugu_df = df[['INDIC REVIEW', 'LABEL']].rename(
            columns={'INDIC REVIEW': 'text', 'LABEL': 'label'}
        )
        
        # Map labels to integers for easier evaluation
        # Normalize label case as the dataset might have mixed case labels
        telugu_df['label'] = telugu_df['label'].str.lower()
        print(f"Unique label values: {telugu_df['label'].unique()}")
        
        # Ensure label is properly mapped to integers
        # Make sure any None/null values are replaced with a valid class
        label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        telugu_df['label'] = telugu_df['label'].map(label_map)
        
        # Replace any remaining NaN values with a default class (e.g., neutral)
        telugu_df['label'] = telugu_df['label'].fillna(2)
        
        # Convert to integer
        telugu_df['label'] = telugu_df['label'].astype(int)
        
        telugu_data.append((split, telugu_df))
    
    # Save to disk
    os.makedirs("tasks_data/telugu_sentiment", exist_ok=True)
    
    for split_name, split_data in telugu_data:
        output_file = f"tasks_data/telugu_sentiment/{split_name}.csv"
        split_data.to_csv(output_file, index=False)
        print(f"Saved {len(split_data)} Telugu examples to {output_file}")

if __name__ == "__main__":
    download_and_filter_indic_sentiment()