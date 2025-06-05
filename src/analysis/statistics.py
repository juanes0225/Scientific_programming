# src/analysis/statistics.py


import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import f_oneway




def anova_test(df: pd.DataFrame, group_col: str, target_col: str) -> dict:
    
    # Collect data per group
    groups = df[group_col]
    group_data = [df[df[group_col] == i][target_col].values for i in groups]

    # Compute F and p-value
    F_stat, p_val = f_oneway(*group_data)

   
    return {
        'F_statistic': F_stat,
        'p_value': p_val        
    }


def compute_fisher_ratio(df: pd.DataFrame, feature_col: str, label_col: str) -> float:
    
    #Compute the Fisher Ratio (Fisher’s discriminant ratio) for a single feature.

    labels = df[label_col].dropna().unique()
    if len(labels) != 2:
        raise ValueError("Fisher Ratio requires exactly two classes.")

    class1, class2 = labels
    data1 = df[df[label_col] == class1][feature_col].dropna()
    data2 = df[df[label_col] == class2][feature_col].dropna()

    mu1, mu2 = data1.mean(), data2.mean()
    var1, var2 = data1.var(ddof=1), data2.var(ddof=1)

    fisher_ratio = ((mu1 - mu2) ** 2) / (var1 + var2)
    return fisher_ratio





def separability_measure(df: pd.DataFrame, feature_cols: list, label_col: str) -> pd.DataFrame:
    
    #Compute Fisher Ratio for each feature in feature_cols as a measure of class separability.

    results = []
    for feat in feature_cols:
        try:
            fr = compute_fisher_ratio(df, feat, label_col)
        except ValueError:
            fr = np.nan
        results.append({'feature': feat, 'fisher_ratio': fr})

    result_df = pd.DataFrame(results).sort_values(by='fisher_ratio', ascending=False).reset_index(drop=True)
    return result_df


def compute_covariance_matrix(data: pd.DataFrame, columns: list, output_path: Path = None) -> np.ndarray:
    
    
    
    data_for_covariance = data[columns].dropna().values
    cov_matrix = np.cov(data_for_covariance.T)

    pd.DataFrame(cov_matrix, index=columns, columns=columns).to_csv(output_path)
            
    return cov_matrix

def compute_eigen_decomposition(cov_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    return eigenvalues, eigenvectors