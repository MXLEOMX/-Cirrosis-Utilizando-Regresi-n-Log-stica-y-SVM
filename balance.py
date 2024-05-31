# balance.py
from sklearn.utils import resample
import pandas as pd

def balance_classes(X, y):
    # Procedemos a combinar las características y la variable objetivo
    train_data = pd.concat([X, y], axis=1)

    # Acá separar las clases
    class_0 = train_data[train_data['Status'] == 0]
    class_1 = train_data[train_data['Status'] == 1]
    class_2 = train_data[train_data['Status'] == 2]

    # Realizar sobremuestreo de las clases minoritarias
    class_1_upsampled = resample(class_1, replace=True, n_samples=len(class_0), random_state=42)
    class_2_upsampled = resample(class_2, replace=True, n_samples=len(class_0), random_state=42)

    # acá combinar las clases sobremuestreadas con la clase mayoritaria
    train_data_upsampled = pd.concat([class_0, class_1_upsampled, class_2_upsampled])

    # finalmente separar las características y la variable objetivo
    X_resampled = train_data_upsampled.drop('Status', axis=1)
    y_resampled = train_data_upsampled['Status']
    
    return X_resampled, y_resampled
