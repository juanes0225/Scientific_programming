# src/visualization/plots.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

def plot_normalized_variables(data: pd.DataFrame, variables: list, time_col: str, output_path: Path = None):

    plt.figure(figsize=(10, 4))
    for var in variables:
        plt.plot(data[time_col], data[var], label=var)
    
    plt.title("Normalized Variables: Temperature, Humidity, Pressure") # Considera hacer este título más dinámico si las variables cambian
    plt.xlabel("Time")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path)
    print(f"Gráfico guardado en: {output_path}")
    plt.close() # Cierra la figura para liberar memoria

def generate_and_save_descriptive_statistics(data: pd.DataFrame, output_dir: Path):
    
    
    
    summary_statistics = data.describe()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Definir la ruta completa del archivo CSV
    output_file_path = output_dir / 'summary_statistics.csv'

    # Guardar la tabla en formato CSV
    summary_statistics.to_csv(output_file_path)
     
    return summary_statistics

def plot_features_boxplot(data: pd.DataFrame, features: list = None, output_path: Path = None):

    plt.figure(figsize=(10, 8))
    sns.boxplot(data=data.iloc[:, 1:-1]) 
    plt.title("Box plot of Features")
    plt.xlabel("Features")
    plt.ylabel("Value")
    plt.tight_layout()

    
    plt.savefig(output_path)
    print(f"Box plot guardado en: {output_path}")
    
    plt.close()

def plot_feature_density(data: pd.DataFrame, features: list, hue_col: str, output_path: Path = None):
 
    # Ajustar el layout de subplots dinámicamente
    num_features = len(features)
    rows = (num_features + 1) // 2 # Calcula las filas necesarias (ceil(num_features / 2))
    cols = 2 if num_features > 1 else 1
    plt.figure(figsize=(5 * cols, 4 * rows)) # Ajusta el tamaño de la figura

    for i, feature in enumerate(features):
        plt.subplot(rows, cols, i + 1)
        sns.kdeplot(data=data, x=feature, hue=hue_col, fill=True, alpha=0.5, linewidth=0.5)
        plt.title(f'Density Plot of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_3d_scatter(data: pd.DataFrame, x_col: str, y_col: str, z_col: str, color_col: str, output_path: Path = None):

    fig = plt.figure(figsize=(10, 8)) # Ajusta el tamaño de la figura 3D
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[x_col], data[y_col], data[z_col], c=data[color_col])
    ax.set_xlabel(x_col) 
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    plt.title(f'3D Scatter Plot: {x_col} vs {y_col} vs {z_col}')
    

    plt.tight_layout()
    plt.savefig(output_path)      
    plt.close()


def plot_pairplot(data: pd.DataFrame, hue_col: str = None, output_path: Path = None):

    
    
    g = sns.pairplot(data, hue=hue_col, palette='Set2', markers=["o", "s", "D"], diag_kind='kde')
    plt.suptitle(f"Pairplot of Features by {hue_col}", y=1.02) # Título ajustado
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Ajusta el layout para el suptitle

    
    g.savefig(output_path) 
    plt.close()

def plot_generic_2d_line(data: pd.DataFrame, x_column: str, y_column: str, output_path: Path = None):
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(data[x_column], data[y_column], label=f"Plot of {y_column} Vs {x_column}", color="red")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f"Plot of {y_column} vs {x_column}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_eigenvectors_2d(data_matrix: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                         column_names: list, output_path: Path = None):
    
    
    plt.figure(figsize=(8, 6))
    plt.scatter(data_matrix[:, 0], data_matrix[:, 1], alpha=0.5, label="Data")
    origin = np.mean(data_matrix, axis=0)
    origin = np.mean(data_matrix, axis=0)
    for i in range(len(eigenvalues)):
        u = eigenvectors[0, i] * np.sqrt(eigenvalues[i])
        v = eigenvectors[1, i] * np.sqrt(eigenvalues[i])
        plt.quiver(origin[0], origin[1], u, v,
                color=['r', 'b'][i % 2], scale=3, label=f"Eigenvector {i+1}")

    plt.xlabel(column_names[0] )
    plt.ylabel(column_names[1] )
    plt.legend()
    plt.title("Eigenvalues and Eigenvectors of the Covariance Matrix")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pca_dimensionality_reduction(transformed_data: np.ndarray, labels: pd.Series, output_path: Path = None):
    
    
    colors = {0: 'blue', 1: 'red'}
    labels_dict = {0: 'No Rain', 1: 'Rain'} 
    plt.figure(figsize=(8, 6))
    
    for label_val in [0, 1]:
        
        idx = labels == label_val
        plt.scatter(transformed_data[idx, 0], transformed_data[idx, 1],
                        color=colors[label_val], label=f"{labels_dict.get(label_val, label_val)}", alpha=0.25)
       

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA - Dimensionality Reduction (Classes Highlighted)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
