import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

def plot_confusion_matrix(y_test, y_pred, model_name, filename):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1", "Class 2"], yticklabels=["Class 0", "Class 1", "Class 2"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.savefig(filename)
    plt.show()

def train_evaluate_models(X_resampled, y_resampled, X, y):
    # Acá primero dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Este es el pipeline para Regresión Logística con escalado
    pipe_log_reg = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])

    # Ahora acá se hace ajuste de hiperparámetros para Regresión Logística
    param_grid_log_reg = {
        'log_reg__penalty': ['l2'],
        'log_reg__C': [0.1, 1, 10],
        'log_reg__solver': ['lbfgs'],
        'log_reg__max_iter': [1000, 2000, 3000]
    }
    grid_log_reg = GridSearchCV(pipe_log_reg, param_grid_log_reg, cv=5, scoring='accuracy')
    grid_log_reg.fit(X_resampled, y_resampled)
    best_log_reg = grid_log_reg.best_estimator_
    y_pred_log_reg_best = best_log_reg.predict(X_test)
    report_log_reg_best = classification_report(y_test, y_pred_log_reg_best)
    print("Mejores Hiperparámetros de Regresión Logística:", grid_log_reg.best_params_)
    print("Rendimiento de Regresión Logística con Mejores Hiperparámetros:\n", report_log_reg_best)
    plot_confusion_matrix(y_test, y_pred_log_reg_best, "Logistic Regression", "confusion_matrix_logistic_regression.png")

    # Este es el pipeline para SVM con escalado
    pipe_svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC())
    ])

    # Seguir con el ajuste de hiperparámetros para SVM
    param_grid_svm = {
        'svm__C': [0.1, 1, 10],
        'svm__kernel': ['linear', 'rbf'],
        'svm__gamma': ['scale', 'auto']
    }
    grid_svm = GridSearchCV(pipe_svm, param_grid_svm, cv=5, scoring='accuracy')
    grid_svm.fit(X_resampled, y_resampled)
    best_svm = grid_svm.best_estimator_
    y_pred_svm_best = best_svm.predict(X_test)
    report_svm_best = classification_report(y_test, y_pred_svm_best)
    print("Mejores Hiperparámetros de SVM:", grid_svm.best_params_)
    print("Rendimiento de SVM con Mejores Hiperparámetros:\n", report_svm_best)
    plot_confusion_matrix(y_test, y_pred_svm_best, "SVM", "confusion_matrix_svm.png")

    # ahora procedo a crear un DataFrame con las predicciones y la verdad real para boxplot
    results = pd.DataFrame({
        "Actual": y_test,
        "Logistic Regression": y_pred_log_reg_best,
        "SVM": y_pred_svm_best
    })

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results)
    plt.title("Distribución de las Predicciones")
    plt.savefig("boxplot_predictions.png")
    plt.show()


