# src/analysis/features.py

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class FeatureExtractor:
    
    def __init__(self, dataframe: pd.DataFrame):
        
        self.df = dataframe.copy() # Work with a copy to avoid modifying the original
        self.fs = 1

    def extract_time_features(self, columns: list) -> pd.DataFrame:   # Complexity O(n)
        
        features = {}
        for col in columns:
            signal = self.df[col].values
            features[col] = {
                'mean': np.mean(signal),
                'variance': np.var(signal),
                'rms': np.sqrt(np.mean(signal**2))
            }
        return pd.DataFrame(features).T 

    def extract_frequency_features(self, columns: list) -> pd.DataFrame:  # Complexity O(n)
     
        features = {}
        
        n_full = len(self.df) 
        freqs = np.fft.rfftfreq(n_full, d=1/self.fs)

        for col in columns:
            
            signal = self.df[col].values 
            fft_vals = np.abs(np.fft.rfft(signal))
            power = fft_vals ** 2

            spectral_centroid = np.sum(freqs * power) / np.sum(power)
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power) / np.sum(power))
            peak_freq = freqs[np.argmax(power)]
            features[col] = {
                'peak_freq': peak_freq,
                'spectral_centroid': spectral_centroid,
                'spectral_bandwidth': spectral_bandwidth
            }
        return pd.DataFrame(features).T if features else pd.DataFrame()

    def extract_distribution_features(self, columns: list) -> pd.DataFrame:  # Complexity O(n)
       
        features = {}
        for col in columns:
            
            signal_series = self.df[col].values
            features[col] = {
                'skewness': signal_series.skew(),
                'kurtosis': signal_series.kurt()
            }
        return pd.DataFrame(features).T 

    def extract_derivative_features(self, columns: list) -> pd.DataFrame:   # Complexity O(n^2)
        
        features = {}
        for col in columns:
            
            signal = self.df[col].values
            
            diff = np.diff(signal)
            features[col] = {
                'mean_abs_change': np.mean(np.abs(diff)),
                'std_derivative': np.std(diff)
            }
        return pd.DataFrame(features).T 
    

    def apply_pca_transformation(data: pd.DataFrame, n_components: int = 2) -> tuple[np.ndarray, pd.Series]:  # Complexity O(n)
        

        # Eliminar la columna datetime 'Hora_PC'
        data_filtered = data.drop(columns=['Hora_PC'])

        # Seleccionar solo columnas numéricas para PCA
        data_numeric = data_filtered.select_dtypes(include=[np.number])
        print(data_filtered.dtypes)
        print(data_numeric.dtypes)

        pca = PCA(n_components=n_components, random_state=42)
        transformed_data = pca.fit_transform(data_numeric)

        return transformed_data


