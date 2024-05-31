# main.py
import pandas as pd
from preprocessing import preprocess_data
from modeling import train_evaluate_models
from balance import balance_classes

# Cargar el dataset
file_path = r'C:\Users\LEONARDO\Downloads\ML T2\cirrhosis.csv'  # Usa el prefijo 'r' para una cadena sin formato
cirrhosis_df = pd.read_csv(file_path)

# Preprocesar los datos
X, y = preprocess_data(cirrhosis_df)

# Balancear las clases
X_resampled, y_resampled = balance_classes(X, y)

# Entrenar y evaluar modelos
train_evaluate_models(X_resampled, y_resampled, X, y)
