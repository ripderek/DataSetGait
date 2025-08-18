#ESTE ARCHIVO A DIFERENCIA DEL PRIMERO => AQUI SE TOMAN MUESTRAS CADA 25 FRAMES Y SOLO SE OBTIENE EL PROMEDIO Y LA DESVIACION
#PARA ENTRENAR EL MODELO
import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict, deque
import math



#----------------------------------------------[VARIABLES GLOBALES]----------------------------------------
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



#nombre Carpeta Participante
participante="JosselynV"
    
#NoControlado
#Controlado
ruta_video = "Participantes/"+participante+"/Controlado/Lateral/1.mov" 


#-----------------------------------------------[FUNCIONES]---------------------------------------------------
def suavizar_landmark(idx, x, y):
    history[idx].append((x, y))
    # Promedio
    x_avg = sum(p[0] for p in history[idx]) / len(history[idx])
    y_avg = sum(p[1] for p in history[idx]) / len(history[idx])
    return x_avg, y_avg

#funcion apra mostrar valores en la misma ventana
def mostar_valores(fondo,variable, etiqueta, x, y):
    cv2.putText(fondo, f'{etiqueta}: {variable:.1f}', (x, y),
    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

def distancia_euclidiana(x_A, x_B, y_A, y_B):
    return math.sqrt((x_A - x_B)**2 + (y_A - y_B)**2)

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



"""
#funcion para guardar los puntos en la bd 
def registrar_puntos_muestra(param1, param2, param3, param4, param5, param6,
                              param7, param8, param9, param10, param11, param12):
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            CALL registrar_puntos_muestra(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
       , (
            param1, param2, param3, param4, param5, param6,
            param7, param8, param9, param10, param11, param12
        ))

        conn.commit() 

        cur.close()
        conn.close()
        print("Puntos registrados")

    except Exception as e:
        return {"status": "error", "mensaje": str(e)}
"""
    

#mejorar el frame con rescalador
def mejorar_frame(frame):
    # 1. Reducir tamaño para acelerar filtrado y detección
    scale_factor = 0.6  # ajustar (0.5 = mitad tamaño)
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

def mostrar_resultados_consola(promedio_32_31,desviacion_32_31,promedio_28_27,desviacion_28_27,promedio_26_25,desviacion_26_25,promedio_31_23,desviacion_31_23,promedio_32_24,desviacion_32_24,promedio_16_12,desviacion_16_12,promedio_15_11,desviacion_15_11,promedio_32_16,desviacion_32_16,promedio_31_15,desviacion_31_15):
    print(f"Resultados:")
    print(f"Distancia 32->31: Promedio = {promedio_32_31}, Desviación = {desviacion_32_31}")
    print(f"Distancia 28->27: Promedio = {promedio_28_27}, Desviación = {desviacion_28_27}")
    print(f"Distancia 26->25: Promedio = {promedio_26_25}, Desviación = {desviacion_26_25}")
    print(f"Distancia 31->23: Promedio = {promedio_31_23}, Desviación = {desviacion_31_23}")
    print(f"Distancia 32->24: Promedio = {promedio_32_24}, Desviación = {desviacion_32_24}")
    print(f"Distancia 16->12: Promedio = {promedio_16_12}, Desviación = {desviacion_16_12}")
    print(f"Distancia 15->11: Promedio = {promedio_15_11}, Desviación = {desviacion_15_11}")
    print(f"Distancia 32->16: Promedio = {promedio_32_16}, Desviación = {desviacion_32_16}")
    print(f"Distancia 31->15: Promedio = {promedio_31_15}, Desviación = {desviacion_31_15}")





#-----------------------------------------------------------[FUNCION PRINCIPAL]-------------------------------
#verdadera funcion:
def obtener_marcha(video_path,nombre_persona, muestraid,videoid):
    cap = cv2.VideoCapture(video_path)
    contador = 0
    contador_video = 1

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
    #tomar muestras cada 25 frames

    #cruce de rodillas 
    cruce_rodillas_indicador = False
    orientacion=1 #por defecto se inicia frontal
    #orientacion 1= frontal, 2= espalda, 3= lateral

    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (ANCHO, ALTO))

        #Mejorar el frame y rescalar para optimizacion
        #frame_mejorado, scale  = mejorar_frame(frame)
        frame_mejorado=frame

        #image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.cvtColor(frame_mejorado, cv2.COLOR_BGR2RGB)

        results = pose.process(image_rgb)
       
        if results.pose_landmarks:

            #h, w, _ = frame.shape
            h, w, _ = frame_mejorado.shape

            h_small, w_small, _ = frame_mejorado.shape
            landmarks = results.pose_landmarks.landmark

            # Coordenadas suavizadas
            xs = []
            ys = []
            for idx, lm in enumerate(landmarks):
                #x_s, y_s = suavizar_landmark(idx, lm.x * w, lm.y * h) #anterior suvaizado

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

            #persona_recortada = frame[min_y:max_y, min_x:max_x]
            persona_recortada = frame_mejorado[min_y:max_y, min_x:max_x]

            persona_normalizada = cv2.resize(persona_recortada, (NORMALIZADO_ANCHO, NORMALIZADO_ALTO))
            fondo_negro = np.zeros_like(persona_normalizada)

            nueva_w, nueva_h = NORMALIZADO_ANCHO, NORMALIZADO_ALTO
            escala_x = nueva_w / (max_x - min_x)
            escala_y = nueva_h / (max_y - min_y)

            # Evaluación de orientación (uso de puntos suavizados)
            x_23 = xs[23]
            x_24 = xs[24]
            x_11 = xs[11]
            x_12 = xs[12]
            x_25 = xs[25]
            x_26 = xs[26]

            diff_caderas = abs(x_23 - x_24)
            diff_hombros = abs(x_11 - x_12)
            centro_horizontal = escala_x // 2

            cruce_caderas = (x_23 < centro_horizontal < x_24) or (x_24 < centro_horizontal < x_23)
            cruce_hombros = (x_11 < centro_horizontal < x_12) or (x_12 < centro_horizontal < x_11)
            cruce_rodillas = (x_25 < centro_horizontal < x_26) or (x_26 < centro_horizontal < x_25)

            # Umbral para considerar "la misma coordenada"
            umbral = 5  #  nivel de precisión deseado
            cruce_rodillas_2 = abs(x_25 - x_26) < umbral 

            #-------------------------------------------PUNTOS DISTANCIAS--------------------------------------
                #32_31
            x1_31 = obtener_escala_x(min_x,escala_x,xs[31])
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

            #nuevo
            #deteccion de la orientacion de la persona Frontal, Espalda, Lateral
            mano_izquierda= int((xs[16] - min_x) * escala_x)
            mano_derecha = int((xs[15] - min_x) * escala_x)
            print(f"mano izquierda: {mano_izquierda} - mano derecha: {mano_derecha}")

            # si la persona al menos cruzo las rodillas una vez entonces esta en lateral evaluar de nuevo cuando pasen los 25 fotogramas
                        #orientacion = "Lateral" if (cruce_caderas or cruce_hombros or cruce_rodillas or cruce_rodillas_2 or
                                        #(diff_caderas < 20 and diff_hombros < 20)) else "Frente o Espalda"
            
            
            #en los primeros 10 fotogramas se evalua la orientacion si esque el cruce de rodillas es false
            if contador<=10 and cruce_rodillas_indicador == False:
                #evaluar la orientacion
                #300 es el punto central
                #si la mano izquierda es menor que 300 y la mano derecha es mayor a 300 entonces esta de frente
                #sino de espaldas
                #if mano_izquierda < 300 and mano_derecha > 300:
                #orientacion = "Frontal" if (mano_izquierda < 300 and mano_derecha > 300) else "Espalda"
                orientacion = 1 if (mano_izquierda < 300 and mano_derecha > 300) else 2

            # en caso de que no se ha detectado cruce de rodillas entonces evaluarlo
            if cruce_rodillas_indicador == False:
                #si esta en false es porque no se ha evaluado la orientacion en lateral
                 if (cruce_caderas or cruce_hombros or cruce_rodillas or cruce_rodillas_2 or (diff_caderas < 20 and diff_hombros < 20)):
                     #orientacion = "Lateral"
                     orientacion = 3
                     cruce_rodillas_indicador= True


            #si la los puntos p32 o p31 se pasan de cierta coordenada entonces resetear el contador a 0 y dejar de tomar datos
            pie_xd = int((ys[32] - min_y) * escala_y)
            pie_xd_2 = int((ys[31] - min_y) * escala_y)

            #si un pie se pasa de la linea entonces no realizar mas calculos
            if pie_xd >= 560 or pie_xd_2>=560:  #540 inicialmente
                #print(f"no se realizan calculos")
                cv2.putText(fondo_negro, f'marcha fuera de rango', (10, 550),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)

            else:
                #reiniciar contadores
                if (contador>=25):
                    #calcular la desviacion y el promedio para poder guardar como muestra    
                    promedio_32_31 = obtener_promedio(vector_distancia_32_31)
                    desviacion_32_31 = obtener_desviacion(vector_distancia_32_31)
                    promedio_28_27 = obtener_promedio(vector_distancia_28_27)
                    desviacion_28_27 = obtener_desviacion(vector_distancia_28_27)
                    promedio_26_25 = obtener_promedio(vector_distancia_26_25)
                    desviacion_26_25 = obtener_desviacion(vector_distancia_26_25)
                    promedio_31_23 = obtener_promedio(vector_distancia_31_23)
                    desviacion_31_23 = obtener_desviacion(vector_distancia_31_23)
                    promedio_32_24 = obtener_promedio(vector_distancia_32_24)
                    desviacion_32_24 = obtener_desviacion(vector_distancia_32_24)
                    # NUEVOS PUNTOS PARA REALIZAR MAS PRUEBAS
                    promedio_16_12 = obtener_promedio(vector_distancia_16_12)
                    desviacion_16_12 = obtener_desviacion(vector_distancia_16_12)
                    promedio_15_11 = obtener_promedio(vector_distancia_15_11)
                    desviacion_15_11 = obtener_desviacion(vector_distancia_15_11)
                    promedio_32_16 = obtener_promedio(vector_distancia_32_16)
                    desviacion_32_16 = obtener_desviacion(vector_distancia_32_16)
                    promedio_31_15 = obtener_promedio(vector_distancia_31_15)
                    desviacion_31_15 = obtener_desviacion(vector_distancia_31_15)

                    #ahora en lugar de guardarlo en un json ahora se debe guardar en la base de datos
                    #registrar_puntos_muestra(videoid,muestraid,promedio_32_31,desviacion_32_31,promedio_28_27,desviacion_28_27,promedio_26_25,desviacion_26_25,promedio_31_23,desviacion_31_23,promedio_32_24,desviacion_32_24)
                    mostrar_resultados_consola(promedio_32_31,desviacion_32_31,promedio_28_27,desviacion_28_27,promedio_26_25,desviacion_26_25,promedio_31_23,desviacion_31_23,promedio_32_24,desviacion_32_24,promedio_16_12,desviacion_16_12,promedio_15_11,desviacion_15_11,promedio_32_16,desviacion_32_16,promedio_31_15,desviacion_31_15)

                    contador=0
                    cruce_rodillas_indicador= False
                    #sumar el contador del video
                    #contador_video += 1
          

            #calcular la orientacion de la persona si esta de frente o lateral 
            #orientacion = "Lateral" if (cruce_caderas or cruce_hombros or cruce_rodillas or cruce_rodillas_2 or
                                        #(diff_caderas < 20 and diff_hombros < 20)) else "Frente o Espalda"

            cv2.putText(fondo_negro, f'Orientacion: {orientacion}', (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(fondo_negro, f'Contador: {contador}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(fondo_negro, f'diff_caderas: {diff_caderas:.1f}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.putText(fondo_negro, f'diff_hombros: {diff_hombros:.1f}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.putText(fondo_negro, f'cruce_rodillas_2: {cruce_rodillas_2:.1f}', (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.putText(fondo_negro, f'{nombre_persona}', (10, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.putText(fondo_negro, 'Presiona Q para salir', (10, 480),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Dibujar conexiones suavizadas
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx not in puntos_rostro and end_idx not in puntos_rostro:
                    x1 = int((xs[start_idx] - min_x) * escala_x)
                    y1 = int((ys[start_idx] - min_y) * escala_y)
                    x2 = int((xs[end_idx] - min_x) * escala_x)
                    y2 = int((ys[end_idx] - min_y) * escala_y)
                    cv2.line(fondo_negro, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Dibujar puntos suavizados
            for idx in range(len(xs)):
                if idx not in puntos_rostro:
                    x = int((xs[idx] - min_x) * escala_x)
                    y = int((ys[idx] - min_y) * escala_y)
                    cv2.circle(fondo_negro, (x, y), 3, (0, 255, 0), -1)
                    cv2.putText(fondo_negro, f'p{idx} ({x},{y})', (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.line(fondo_negro, (nueva_w // 2, 0), (nueva_w // 2, nueva_h), (255, 255, 0), 2)
            cv2.line(fondo_negro, (0, nueva_h // 2), (nueva_w, nueva_h // 2), (255, 0, 0), 2)

            #linea limite para los pies, si se pasa de esta linea entonces no tomar datos
            y_linea = int(nueva_h * 0.96) # estaba en 0.9 pero se amplio la zona de reconocimiento
            cv2.line(fondo_negro, (0, y_linea), (nueva_w, y_linea), (255, 0, 0), 2)

            #contador ++ para el eye X
            contador += 1
            print(f'contador: {contador}')
            #cv2.putText(fondo_negro, f'contador: {contador}', (700, 500),
                        #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
            

            #comentar para que trabaje en segundo plano
            cv2.imshow('Solo Esqueleto Normalizado', fondo_negro)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #video_path,nombre_persona, muestraid,videoid
    obtener_marcha(ruta_video,"ejemplo",1,1)