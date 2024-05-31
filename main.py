# main.py
import pandas as pd
from preprocessing import preprocess_data
from modeling import train_evaluate_models
from balance import balance_classes

# Realizar la cargar del dataset
file_path = r'C:\Users\LEONARDO\Downloads\ML T2\cirrhosis.csv'  # Usa el prefijo 'r' para una cadena sin formato
cirrhosis_df = pd.read_csv(file_path)

# Acá realizamos el preprocesamiento delos datos
X, y = preprocess_data(cirrhosis_df)

# Luego balanceamos las clases
X_resampled, y_resampled = balance_classes(X, y)

# Finalmente procedemos a entrenar los modelos y los evaluamos con sus resultados específicos 
train_evaluate_models(X_resampled, y_resampled, X, y)
