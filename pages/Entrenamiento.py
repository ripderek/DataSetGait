import streamlit as st
import os
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load
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

#STREAMLIT
# Set the title for the Streamlit app
st.set_page_config(
    page_title="Entrenamiento",
    layout="wide",  # <--- esto hace que el contenedor ocupe todo el ancho
    initial_sidebar_state="auto"
)
# Título de la app
st.title("Entrenamiento de modelos")

iniciar_entrenamiento = st.button("Iniciar entrenamiento de modelos")

#st.subheader("Entrenamiento de MLP (Multi-Layer Perceptron)")
#tiempo_MLP = st.empty()
#Accuracy = st.empty()
#Classification_Report = st.empty()

st.subheader("Entrenamiento de RF (Random Forest)")
tiempo_RF_1 = st.empty()

# Variable global para acumular logs
log_texto_RF = ""

def log_rf(msg):
    """Acumula logs y actualiza el placeholder en pantalla"""
    global log_texto_RF, tiempo_RF_1
    log_texto_RF += msg + "\n"  # agregar mensaje con salto de línea
    tiempo_RF_1.text(log_texto_RF)  # reescribir todo el log

def EntrenamientoMLP_f(numero):
    #if iniciar_entrenamiento:
        #tiempo_MLP.write(f"MLP-> Iniciando entrenamiento {numero}")
        # Paso 1: Obtener JSON desde PostgreSQL
        resultado = ec.ObtenerDatosEntrenamiento(numero)  # El JSON como dict/list
        datos = resultado

        X = []
        y = []

        # Paso 2: Convertir muestras en vectores
        for muestra in datos:
            vector = []
            puntos = muestra["puntos"]

            for clave in sorted(puntos.keys()):
                vector.append(puntos[clave]["promedio"])
                vector.append(puntos[clave]["desviacion"])

            X.append(vector)
            y.append(muestra["persona"])

        X = np.array(X)
        y = np.array(y)

        # Paso 3: Codificar etiquetas
        os.makedirs("MODELOS", exist_ok=True)
        #encoder_path = f"MODELOS/MLP/label_encoder_All.joblib"
        #model_path = f"MODELOS/MLP/mlp_model_ALL.h5"
        encoder_path = f"MODELOS/MLP/label_encoder_{numero}.joblib"
        model_path = f"MODELOS/MLP/mlp_model_{numero}.h5"

        #if os.path.exists(encoder_path):
            # Cargar encoder existente
            #label_encoder = load(encoder_path)
            #y_encoded = label_encoder.transform(y)
            #print("Encoder cargado desde archivo existente.")
            #tiempo_MLP.write(f"Encoder cargado desde archivo existente.")
        #else:
        # Crear nuevo encoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        dump(label_encoder, encoder_path)
        #print("Encoder creado y guardado.")
        #tiempo_MLP.write(f"MLP-> Encoder creado y guardado {numero}. ")

        y_categorical = to_categorical(y_encoded)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42
        )

        # Paso 4: Crear o cargar modelo
        #if os.path.exists(model_path):
            #modelo = load_model(model_path)
            #print("Modelo existente cargado, continuará el entrenamiento.")
        #else:
        input_dim = X.shape[1]
        output_dim = y_categorical.shape[1]
        modelo = Sequential()
        modelo.add(Dense(128, input_dim=input_dim, activation='relu'))
        modelo.add(Dropout(0.3))
        modelo.add(Dense(64, activation='relu'))
        modelo.add(Dense(output_dim, activation='softmax'))
        modelo.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])
        #tiempo_MLP.write(f"MLP-> Nuevo modelo MLP creado {numero}.")
        # Paso 5: Entrenamiento
        inicio = time.time()
        modelo.fit(X_train, y_train, epochs=30, batch_size=16, verbose=1)
        fin = time.time()
        duracion = fin - inicio

        # Evaluación
        y_pred = np.argmax(modelo.predict(X_test), axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        #Accuracy.write(f"Accuracy: {accuracy_score(y_test_labels, y_pred)}")
        #Classification_Report.write(f"Classification Report: {classification_report(y_test_labels, y_pred, target_names=label_encoder.classes_)}")

        # Guardar modelo
        modelo.save(model_path)
        dump(label_encoder, encoder_path)
        #tiempo_MLP.write(f"MLP->Modelo {numero} entrenado y guardado exitosamente. Tiempo de entrenamiento: {duracion:.2f} segundos")
        #print(f"Modelo entrenado y guardado exitosamente. Tiempo de entrenamiento: {duracion:.2f} segundos")


#funcion para el entrenamiento de RF -> random forest
def EntrenamientoRF(numero):
    # Paso 1: Obtener JSON desde PostgreSQL
    resultado = ec.ObtenerDatosEntrenamiento(numero)  # El JSON como dict/list
    datos = resultado

    X = []
    y = []

    #tiempo_RF_1.write(f"RF -> Iniciando entrenamiento Modelo_{numero}")
    log_rf(f"RF -> Iniciando entrenamiento Modelo_{numero}")
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
    log_rf(f"Modelo entrenado y guardado exitosamente como: {nombre_archivo}")
    log_rf(f"Tiempo de entrenamiento: {duracion:.2f} segundos")
    log_rf(f"Mejores parámetros: {mejor_params}")




if __name__ == "__main__":
    if iniciar_entrenamiento:
        #EntrenamientoMLP_f(1)
        #EntrenamientoMLP_f(2)
        #EntrenamientoMLP_f(3)
        EntrenamientoRF(1)
        EntrenamientoRF(2)
        EntrenamientoRF(3) 
