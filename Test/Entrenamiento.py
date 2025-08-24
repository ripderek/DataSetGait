
import os
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from db import get_connection
import controllers.EntrenamientoController as ec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Modelos
Sequential = tf.keras.Sequential
load_model = tf.keras.models.load_model

# Capas
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout

# Utilidades
to_categorical = tf.keras.utils.to_categorical

# Variable global para acumular logs
log_texto_RF = ""




#funcion para el entrenamiento de RF -> random forest
def EntrenamientoRF(numero):
    # Paso 1: Obtener JSON desde PostgreSQL
    resultado = ec.ObtenerDatosEntrenamiento(numero)  # El JSON como dict/list
    datos = resultado
    X = []
    y = []

    #tiempo_RF_1.write(f"RF -> Iniciando entrenamiento Modelo_{numero}")
    print(f"RF -> Iniciando entrenamiento Modelo_{numero}")
    # Paso 2: Convertir muestras en vectores
    for muestra in datos:
        vector = []
        puntos = muestra["puntos"]

        for clave in sorted(puntos.keys()):
            vector.append(puntos[clave]["promedio"])
            vector.append(puntos[clave]["desviacion"])

        X.append(vector)
        y.append(muestra["persona"])

    # Definir Random Forest y parámetros a optimizar
    modelo = RandomForestClassifier(random_state=42)
    
    # Definir el espacio de búsqueda de hiperparámetros
    param_dist = {
        "n_estimators": randint(50, 500),     # Número de árboles en el bosque (más árboles = más robusto pero más lento)
        "max_depth": [None, 10, 20, 30, 40],  # Profundidad máxima de cada árbol (None = crece hasta que no puede más)
        "min_samples_split": randint(2, 20),  # Mínimo de muestras necesarias para dividir un nodo
        "min_samples_leaf": randint(1, 10),   # Mínimo de muestras en una hoja (reduce overfitting si es >1)
        "max_features": ["sqrt", "log2", None], # Número de features consideradas en cada split (sqrt = default para clasificación)
        "bootstrap": [True, False],           # Si se hace muestreo con reemplazo (True) o no (False)
    }

    # RandomizedSearchCV en lugar de GridSearchCV
    random_search = RandomizedSearchCV(
        estimator=modelo,        # Modelo base que se ajusta
        param_distributions=param_dist,  # Espacio de parámetros a muestrear
        n_iter=50,               # Número de combinaciones aleatorias a probar (ajustar según recursos)
        cv=3,                    # Validación cruzada con 3 folds
        verbose=2,               # Nivel de detalle en consola (0 = nada, 2 = más información)
        random_state=42,         # Reproducibilidad
        n_jobs=-1                # Usar todos los núcleos disponibles del procesador
    )

    # Paso 3: Entrenar modelo
    inicio = time.time()
    #modelo = RandomForestClassifier()
    #modelo.fit(X, y)
    #grid_search.fit(X, y)
    # Entrenar la búsqueda con los datos
    random_search.fit(X, y)
    fin = time.time()

    mejor_modelo = random_search.best_estimator_
    mejor_params = random_search.best_params_

    duracion = fin - inicio

    # Paso 4: Crear carpeta si no existe
    os.makedirs("MODELOS", exist_ok=True)

    # Paso 5: Guardar modelo con timestamp único
    #timestamp = datetime.now().strftime("%d%m%Y%H%M%S")
    #nombre_modelo = f"modelo_{timestamp}"
    nombre_archivo = f"MODELOS/RF/modelo_{numero}.joblib"
    dump(mejor_modelo, nombre_archivo)

    #editar en la bd e insertar el nuevo nombre del modelo entrenado
    #tiempo_RF_1.write(f"Modelo entrenado y guardado exitosamente como: {nombre_archivo} Tiempo de entrenamiento: {duracion:.2f} segundos")
    print(f"Modelo entrenado y guardado exitosamente como: {nombre_archivo}")
    print(f"Tiempo de entrenamiento: {duracion:.2f} segundos")
    print(f"Mejores parámetros: {mejor_params}")




if __name__ == "__main__":
    EntrenamientoRF(1)
    EntrenamientoRF(2)
    EntrenamientoRF(3) 
