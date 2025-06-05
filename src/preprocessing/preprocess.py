# src/preprocessing/preprocess.py

import pandas as pd

def load_data(filepath):
    #Load dataset from a CSV file.
   
    return pd.read_csv(filepath)

def convert_datetime(df, column_name="Hora_PC"):  
    #Convert specified column to datetime format.

    if column_name in df.columns:
        df[column_name] = pd.to_datetime(df[column_name])
    return df

def drop_duplicates(df):
    
    #Drop duplicate rows.
    return df.drop_duplicates()

def scale_features(df):

    #Standardize the features (z-score normalization)

    df_scaled = df.copy()
    df_scaled.iloc[:, 1:-1] = df_scaled.iloc[:, 1:-1].transform(lambda x: (x - x.mean()) / x.std())
    return df_scaled

def process_data(input_path='data/raw/data.csv',output_path='data/processed/cleaned_data.csv'):
    
    #Full preprocessing.

    df = load_data(input_path)
    df = convert_datetime(df, column_name="Hora_PC")
    df = drop_duplicates(df)
    df_scaled = scale_features(df)

    df_scaled.to_csv(output_path, index=False)
    print(f"Cleaned and scaled data saved to: {output_path}")

    return df_scaled

