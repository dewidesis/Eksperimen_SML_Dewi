import pandas as pd
import re
import sys
import os

def clean_column_names(df):
    df.columns = [re.sub(r'[^\w]', '', re.sub(r'[\s\-]+', '_', col.strip())).lower()
                  for col in df.columns]
    return df

def encode_gender(df):
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'M': 0, 'F': 1}).fillna(-1).astype(int)
    return df

def encode_lung_cancer(df):
    if 'lung_cancer' in df.columns:
        df['lung_cancer'] = df['lung_cancer'].map({'YES': 1, 'NO': 0}).fillna(-1).astype(int)
    return df

def clean_binary_columns(df, columns):
    valid_cols = [c for c in columns if c in df.columns]
    df[valid_cols] = df[valid_cols].replace({1: 0, 2: 1})
    return df

def drop_duplicates(df):
    return df.drop_duplicates()

def preprocess(df):
    df = clean_column_names(df)
    df = drop_duplicates(df)
    df = encode_gender(df)
    df = encode_lung_cancer(df)

    binary_columns = ['smoking', 'yellow_fingers', 'anxiety', 'peer_pressure',
                      'chronic_disease', 'fatigue', 'allergy', 'wheezing',
                      'alcohol_consuming', 'coughing', 'shortness_of_breath',
                      'swallowing_difficulty', 'chest_pain']
    df = clean_binary_columns(df, binary_columns)
    return df

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python automate.py <input_csv_path> <output_csv_path>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Read and process
    df = pd.read_csv(input_file)
    df_cleaned = preprocess(df)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_cleaned.to_csv(output_file, index=False)
    print(f"Preprocessing complete. Cleaned data saved to: {output_file}")