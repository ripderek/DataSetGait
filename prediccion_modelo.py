import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict, deque
import math
import pandas as pd
import numpy as np
from joblib import load
from sklearn.ensemble import RandomForestClassifier

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Índices de los puntos del rostro en MediaPipe Pose
#hasta el punto 9 son del rostro, desde el 17 son puntos de las manos que de momento son innecesarios
# el punto 0 es el centro de la cabeza para tomar como referencia el medio
# 18 y 19 son puntos  mano
#29 y 30 talones
puntos_rostro = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  17,18, 19, 20, 22, 21 ,29, 30}

#resolucion de video
ANCHO = 960
ALTO = 540

#resolucion para normalizar los videos sin importar la resulicion original de los videos
#NORMALIZADO_ANCHO = 256
#NORMALIZADO_ALTO = 256

NORMALIZADO_ANCHO = 600
NORMALIZADO_ALTO = 600

# Historial para suavizado
history = defaultdict(lambda: deque(maxlen=10))  # Últimos 5 frames

padding = 0.2 #margen del cuadro con respecto a la persona identificada

#DELAY PARA VER LOS VIDEOS
delay =1

tamano_texto_identificacion =4.5

precision_identificacion = 60  #incialmente en 70 ----> si es neceario colocarlo en 0

modelo_entrenado = "Modelos/modelo_12082025175740.joblib"

#nombre Carpeta Participante
participante="AdrianJ2"
    
#NoControlado
#Controlado
ruta_video = "Participantes/"+participante+"/Controlado/Lateral/3.mp4" 

#estado para contar los VP -> Verdaderos Positivos
VP = {"contador": 0}
#estado para contar los FP -> Falsos Positivos
FP = {"contador": 0}
#estado para contrar los PI -> precision_identificacion
PI = {"contador": 0}

#VP["contador"] += 1
#--------------------------------------------------------------------------------------------------------------------

#suavizar el fecto de distorcion
def suavizar_landmark(idx, x, y):
    history[idx].append((x, y))
    # Promedio
    x_avg = sum(p[0] for p in history[idx]) / len(history[idx])
    y_avg = sum(p[1] for p in history[idx]) / len(history[idx])
    return x_avg, y_avg

def calcular_distancias(puntoA_X,puntoA_Y, puntoB_X, puntoB_Y):
    return  math.sqrt((puntoB_X - puntoA_X)**2 + (puntoB_Y - puntoA_Y)**2)

#funcion apra mostrar valores en la misma ventana
def mostar_valores_backup(fondo,variable, etiqueta, x, y):
    cv2.putText(fondo, f'{etiqueta}: {variable:.1f}', (x, y),
    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


def mostar_valores(imagen, valor, texto, x, y):
    cv2.putText(imagen, f"{texto}: {valor:.2f}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def distancia_euclidiana(x_A, x_B, y_A, y_B):
    return math.sqrt((x_A - x_B)**2 + (y_A - y_B)**2)


#funcion para mejorar los frames para ayudar a la deteccion de la persona
def mejorar_frame_solo_filtro(frame):
    # 1. Reducir tamaño para acelerar filtrado
    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # 2. Reducción de ruido más rápida que bilateral
    small = cv2.GaussianBlur(small, (3, 3), 0)

    # 3. Mejorar contraste con CLAHE ligero
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    small = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4. Nitidez ligera
    kernel_sharpening = np.array([
        [0, -0.25, 0],
        [-0.25, 2, -0.25],
        [0, -0.25, 0]
    ])
    small = cv2.filter2D(small, -1, kernel_sharpening)

    # 5. Volver al tamaño original
    frame = cv2.resize(small, (frame.shape[1], frame.shape[0]))

    return frame

#mejorar el frame pero con rescalador
def mejorar_frame(frame):
    # 1. Reducir tamaño para acelerar filtrado y detección
    scale_factor = 0.8  # ajustar (0.5 = mitad tamaño)
    small = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

    # 2. Reducción de ruido rápida
    small = cv2.GaussianBlur(small, (3, 3), 0)

    # 3. Mejorar contraste
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=0.6, tileGridSize=(1, 1))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    small = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4. Nitidez ligera
    #kernel_sharpening = np.array([
       # [0, -0.25, 0],
      #  [-0.25, 2, -0.25],
     #   [0, -0.25, 0]
    #])
    kernel_sharpening = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    small = cv2.filter2D(small, -1, kernel_sharpening)

    # 5. Devuelve el frame reducido y mejorado (NO vuelvas a tamaño original)
    return small, scale_factor




def visualizar_todo(video_path):
    cap = cv2.VideoCapture(video_path)
    contador = 0
    z_score_32_31=0
    #vectores de distancias
    vector_distancia_32_31 = []
    vector_distancia_28_27 =[]
    vector_distancia_26_25 =[]
    vector_distancia_31_23 =[]
    vector_distancia_32_24 =[]
    #nuevos vectores 
    vector_distancia_16_12 =[]
    vector_distancia_15_11 =[]
    vector_distancia_32_16 =[]
    vector_distancia_31_15 =[]

    persona_identificada = "No identificada"
    persona_identificada2 = "No identificada"
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (ANCHO, ALTO))

        #aplicacion de mejoras del frame 
        frame_mejorado, scale  = mejorar_frame(frame)
        #frame_mejorado = frame

        #image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.cvtColor(frame_mejorado, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            h, w, _ = frame_mejorado.shape 
            h_small, w_small, _ = frame_mejorado.shape
            landmarks = results.pose_landmarks.landmark

            xs, ys = [], []
            for idx, lm in enumerate(landmarks):
                #x_s, y_s = suavizar_landmark(idx, lm.x * w, lm.y * h) anterior solo utilizando la resolucion oriignal


                # Escalar la posición del landmark a tamaño original
                x_orig = lm.x * w_small * (w / w_small)  # Simplifica a lm.x * w_orig
                y_orig = lm.y * h_small * (h / h_small)  # Simplifica a lm.y * h_orig
                # Usar suavizador con coordenadas en tamaño original
                x_s, y_s = suavizar_landmark(idx, x_orig, y_orig)

                xs.append(x_s)
                ys.append(y_s)

            min_x, max_x = int(min(xs)), int(max(xs))
            min_y, max_y = int(min(ys)), int(max(ys))

            padding_x = int((max_x - min_x) * padding)
            padding_y = int((max_y - min_y) * padding)

            min_x = max(0, min_x - padding_x)
            max_x = min(w, max_x + padding_x)
            min_y = max(0, min_y - padding_y)
            max_y = min(h, max_y + padding_y)

            # Dibujar recuadro
            cv2.rectangle(frame_mejorado, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

            # Dibujar esqueleto omitiendo puntos del rostro
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx not in puntos_rostro and end_idx not in puntos_rostro:
                    x1, y1 = int(xs[start_idx]), int(ys[start_idx])
                    x2, y2 = int(xs[end_idx]), int(ys[end_idx])
                    cv2.line(frame_mejorado, (x1, y1), (x2, y2), (0, 0, 255), 2)

            for idx, (x, y) in enumerate(zip(xs, ys)):
                if idx not in puntos_rostro:
                    cv2.circle(frame_mejorado, (int(x), int(y)), 3, (0, 255, 0), -1)

            # Normalización (solo para cálculos)
            #persona_recortada = frame[min_y:max_y, min_x:max_x]
            #persona_normalizada = cv2.resize(persona_recortada, (NORMALIZADO_ANCHO, NORMALIZADO_ALTO))

            nueva_w, nueva_h = NORMALIZADO_ANCHO, NORMALIZADO_ALTO
            escala_x = nueva_w / (max_x - min_x)
            escala_y = nueva_h / (max_y - min_y)

            # Medidas y orientación
            x_23, x_24 = xs[23] * escala_x, xs[24] * escala_x
            x_11, x_12 = xs[11] * escala_x, xs[12] * escala_x
            x_25, x_26 = xs[25] * escala_x, xs[26] * escala_x
            centro_horizontal = escala_x // 2

            cruce_caderas = (x_23 < centro_horizontal < x_24) or (x_24 < centro_horizontal < x_23)
            cruce_hombros = (x_11 < centro_horizontal < x_12) or (x_12 < centro_horizontal < x_11)
            cruce_rodillas = (x_25 < centro_horizontal < x_26) or (x_26 < centro_horizontal < x_25)
            cruce_rodillas_2 = abs(x_25 - x_26) < 5

            diff_caderas = abs(x_23 - x_24)
            diff_hombros = abs(x_11 - x_12)

            orientacion = "Lateral" if (cruce_caderas or cruce_hombros or cruce_rodillas or cruce_rodillas_2 or
                                        (diff_caderas < 20 and diff_hombros < 20)) else "Frente o Espalda"

            # Mostrar orientación sobre el frame
            cv2.putText(frame_mejorado, f'Orientacion: {orientacion}', (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


            #-------------------------------------------PUNTOS DISTANCIAS--------------------------------------
                #32_31
            x1_31 =obtener_escala_x(min_x,escala_x,xs[31])
            y1_31 = obtener_escala_y(min_y,escala_y,ys[31]) 

            x1_32 = obtener_escala_x(min_x,escala_x,xs[32])
            y1_32 = obtener_escala_y(min_y,escala_y,ys[32]) 

            distancia_32_21= 0
            distancia_32_21 = distancia_euclidiana(x1_32,x1_31,y1_32,y1_31)
            vector_distancia_32_31.append(distancia_32_21)

                #28_27
            x1_27 = obtener_escala_x(min_x,escala_x,xs[27])
            y1_27 = obtener_escala_y(min_y,escala_y,ys[27]) 

            x1_28 = obtener_escala_x(min_x,escala_x,xs[28])
            y1_28 = obtener_escala_y(min_y,escala_y,ys[28]) 

            distancia_28_27= 0
            distancia_28_27 = distancia_euclidiana(x1_28,x1_27,y1_28,y1_27)
            vector_distancia_28_27.append(distancia_28_27)

                #26_25
            x1_25 = obtener_escala_x(min_x,escala_x,xs[25])
            y1_25 = obtener_escala_y(min_y,escala_y,ys[25])

            x1_26 = obtener_escala_x(min_x,escala_x,xs[26])
            y1_26 = obtener_escala_y(min_y,escala_y,ys[26])

            distancia_26_25 =0
            distancia_26_25 =distancia_euclidiana(x1_26,x1_25,y1_26,y1_25)
            vector_distancia_26_25.append(distancia_26_25)

                #31_23
            x1_23 = obtener_escala_x(min_x,escala_x,xs[23])
            y1_23 = obtener_escala_y(min_y,escala_y,ys[23])

            x1_31 = obtener_escala_x(min_x,escala_x,xs[31])
            y1_31 = obtener_escala_y(min_y,escala_y,ys[31])

            distancia_31_23 =0
            distancia_31_23 =distancia_euclidiana(x1_31,x1_23,y1_31,y1_23)
            vector_distancia_31_23.append(distancia_31_23)

                #32_24
            x1_24 = obtener_escala_x(min_x,escala_x,xs[24])
            y1_24 = obtener_escala_y(min_y,escala_y,ys[24])

            x1_32 = obtener_escala_x(min_x,escala_x,xs[32])
            y1_32 = obtener_escala_y(min_y,escala_y,ys[32])

            distancia_32_24 =0
            distancia_32_24 =distancia_euclidiana(x1_32,x1_24,y1_32,y1_24)
            vector_distancia_32_24.append(distancia_32_24)

            #nuevos puntos 
            #16_12
            x1_12 = obtener_escala_x(min_x,escala_x,xs[12])
            y1_12 = obtener_escala_y(min_y,escala_y,ys[12])
            x1_16 = obtener_escala_x(min_x,escala_x,xs[16])
            y1_16 = obtener_escala_y(min_y,escala_y,ys[16])

            distancia_16_12 = 0
            distancia_16_12 = distancia_euclidiana(x1_16,x1_12,y1_16,y1_12)
            vector_distancia_16_12.append(distancia_16_12)

            #15_11
            x1_11 = obtener_escala_x(min_x,escala_x,xs[11])
            y1_11 = obtener_escala_y(min_y,escala_y,ys[11])
            x1_15 = obtener_escala_x(min_x,escala_x,xs[15])
            y1_15 = obtener_escala_y(min_y,escala_y,ys[15])

            distancia_15_11 = 0
            distancia_15_11 = distancia_euclidiana(x1_15,x1_11,y1_15,y1_11)
            vector_distancia_15_11.append(distancia_15_11)

            #32_16
            x1_16 = obtener_escala_x(min_x,escala_x,xs[16])
            y1_16 = obtener_escala_y(min_y,escala_y,ys[16])
            x1_32 = obtener_escala_x(min_x,escala_x,xs[32])
            y1_32 = obtener_escala_y(min_y,escala_y,ys[32])
            distancia_32_16 = 0
            distancia_32_16 = distancia_euclidiana(x1_32,x1_16,y1_32,y1_16)
            vector_distancia_32_16.append(distancia_32_16)

            #31_15
            x1_15 = obtener_escala_x(min_x,escala_x,xs[15])
            y1_15 = obtener_escala_y(min_y,escala_y,ys[15])
            x1_31 = obtener_escala_x(min_x,escala_x,xs[31])
            y1_31 = obtener_escala_y(min_y,escala_y,ys[31])

            distancia_31_15 = 0
            distancia_31_15 = distancia_euclidiana(x1_31,x1_15,y1_31,y1_15)
            vector_distancia_31_15.append(distancia_31_15)

            #--------------------------------------------------------------------------------------------------

            
            #si la los puntos p32 o p31 se pasan de cierta coordenada entonces resetear el contador a 0 y dejar de tomar datos
            pie_xd = int((ys[32] - min_y) * escala_y)
            pie_xd_2 = int((ys[31] - min_y) * escala_y)

            if pie_xd >= 560 or pie_xd_2>=560:
                #print(f"no se realizan calculos")
                contador=0
                vector_distancia_32_31.clear()
                persona_identificada = "No identificada"
                cv2.putText(frame_mejorado, f'marcha fuera de rango', (10, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
            else:
                tamano_texto = 0.5
                #texto_c = (max_y/min_y)/tamano_texto_identificacion
                #if texto_c > 1.5:
                    #tamano_texto = 0.9
                #else:
                    #tamano_texto=texto_c

                cv2.putText(frame_mejorado, f'{persona_identificada}', (int(xs[0]-40), int(ys[0])-30),
                        cv2.FONT_HERSHEY_SIMPLEX, tamano_texto, (255, 255, 255), 2)
                #cv2.putText(frame, f"z_score_32_31: {z_score_32_31}", (10, 250),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.98, (255, 255, 255), 1)
                #print(f"{distancia}")
                #reiniciar el contador cuando se considere que hubo una marcha completa de 25s
                if (contador==25):
                    #--------------------------------------------------------------------
                    #Obtener promedios y desviaciones
                    #   32_31
                    #promedio_32_31 = obtener_promedio(vector_distancia_32_31)
                    #desviacion_32_31 = obtener_desviacion(vector_distancia_32_31)
                    #print(f"32_31 => {promedio_32_31},{desviacion_32_31}")
                    #   28_27
                    #promedio_28_27= obtener_promedio(vector_distancia_28_27)
                    #desviacion_28_27 = obtener_desviacion(vector_distancia_28_27)
                    #print(f"28_27 => {promedio_28_27},{desviacion_28_27}")
                    #   26_25
                    #promedio_26_25= obtener_promedio(vector_distancia_26_25)
                    #desviacion_26_25 = obtener_desviacion(vector_distancia_26_25)
                    #print(f"26_25 => {promedio_26_25},{desviacion_26_25}")
                    #   31_23
                    #promedio_31_23= obtener_promedio(vector_distancia_31_23)
                    #desviacion_31_23 = obtener_desviacion(vector_distancia_31_23)
                    #print(f"31_23 => {promedio_31_23},{desviacion_31_23}")
                    #   32_24
                    #promedio_32_24= obtener_promedio(vector_distancia_32_24)
                    #desviacion_32_24 = obtener_desviacion(vector_distancia_32_24)
                    #print(f"32_24 => {promedio_32_24},{desviacion_32_24}")
    
                    #--------------------------------------------------------------------
                    #print("--------------------------------------------------------")
                    #llamar a la funcion de prediccion
                    vectores = {
                        "26_25": vector_distancia_26_25,
                        "28_27": vector_distancia_28_27,
                        "31_23": vector_distancia_31_23,
                        "32_24": vector_distancia_32_24,
                        "32_31": vector_distancia_32_31,
                        "16_12": vector_distancia_16_12,
                        "15_11": vector_distancia_15_11,
                        "32_16": vector_distancia_32_16,
                        "31_15": vector_distancia_31_15
                    }
                    prediccion,probabilidad_predicha, probabilidades = predecir_persona_desde_vectores(vectores)
                    #print(f"prediccion=>{prediccion}")
                    #si la probabilidad es igual o mas alta que la precision de probabilidad entonces alli si afirmar a la persona identificada

                    if probabilidad_predicha >= precision_identificacion: 
                        persona_identificada = f"{prediccion} - {probabilidad_predicha}%"
                        persona_identificada2 = prediccion
                        print(f"Persona identificada > {precision_identificacion} >: {prediccion}")
                        #Calcular el PI    
                        #if prediccion == participante:
                            #PI["contador"] += 1
                            #print(f"(PI)=+1")
                    
                    #Calcular el PI    
                    if persona_identificada2 == participante:
                        PI["contador"] += 1
                        print(f"Se mantiene el (PI)=+1")

                    #Calcular los VP, FP    
                    if prediccion == participante:
                        VP["contador"] += 1
                        #PI["contador"] += 1
                        print(f"(VP)=+1")
                        #print(f"(PI)=+1")
                    else:
                        FP["contador"] += 1
                        print(f"(FP)=+1")

                    #limpiar los vectores
                    vector_distancia_32_31.clear()
                    vector_distancia_28_27.clear()
                    vector_distancia_26_25.clear()
                    vector_distancia_31_23.clear()
                    vector_distancia_32_24.clear()
                    contador=0

            # Incrementar contador
            contador += 1
            cv2.putText(frame_mejorado, f'contador: {contador}', (700, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 

            # Mostrar imagen final directamente
            cv2.imshow('Visualizacion Completa', frame_mejorado)

        else:
            cv2.imshow('Visualizacion Completa', frame_mejorado)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def obtener_escala_x(min_x,escala_x,punto):
    return int((punto - min_x) * escala_x)

def obtener_escala_y(min_y,escala_y,punto):
    return int((punto - min_y) * escala_y)

def obtener_promedio(vector_distancia):
    return sum(vector_distancia) / len(vector_distancia)

def obtener_desviacion(vector_distancia):
    promedio = obtener_promedio(vector_distancia)
    varianza = sum((x - promedio) ** 2 for x in vector_distancia) / len(vector_distancia)
    desviacion = math.sqrt(varianza)
    return desviacion


def predecir_persona_desde_vectores(vectores_distancia: dict):
    """
    vectores_distancia: diccionario con claves tipo "32_31" y valores tipo lista con las 25 distancias
    """
    from joblib import load

    # Cargar modelo
    modelo = load(modelo_entrenado)

    # Construir el vector de entrada ordenadamente
    vector = []
    for clave in sorted(vectores_distancia.keys()):
        distancias = vectores_distancia[clave]
        promedio = obtener_promedio(distancias)
        desviacion = obtener_desviacion(distancias)
        #print(f"{clave} => {promedio}, {desviacion}")
        vector.extend([promedio, desviacion])

    # Realizar la predicción
    prediccion = modelo.predict([vector])[0]
    probabilidades = modelo.predict_proba([vector])[0]

    # Mostrar resultados
    print(f"\n Persona predicha: {prediccion}")
    print(" Probabilidades:")
    for persona, prob in zip(modelo.classes_, probabilidades):
        print(f"- {persona}: {prob * 100:.2f}%")

    # Obtener la probabilidad correspondiente a la persona predicha
    indice = list(modelo.classes_).index(prediccion)
    probabilidad_predicha = round(probabilidades[indice] * 100, 2)

    return prediccion, probabilidad_predicha, dict(zip(modelo.classes_, [round(p * 100, 2) for p in probabilidades]))



if __name__ == "__main__":
    visualizar_todo(ruta_video)
    #mostrar los resultados en base los VP y FP
    print(f"\nResultados de la predicción:--------------------------------------------------------------")
    print(f"Resultados Verdaderos Positivos (VP)= {VP['contador']}")
    print(f"Resultados Falsos Positivos (FP)= {FP['contador']}")
    print(f"Resultados de Precisión de Identificación (PI)= {PI['contador']}")
    suma = PI['contador'] + VP['contador']
    print(f"Resultados de Precisión de Identificación (PI) + (VP)= {suma}")
    print("----------------------------------------------------------------------------------------------")
    print(f"PC-> {VP['contador'] / (VP['contador'] + FP['contador']) * 100:.2f}%")
    print("-----CON PI-------")
    print(f"PC_I-> {suma / (suma + FP['contador']) * 100:.2f}%")

