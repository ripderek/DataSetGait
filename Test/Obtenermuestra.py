import cv2
import mediapipe as mp
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config
import controllers.MuestrasController as sv
import numpy as np
import pandas as pd

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils



reproducir_flag = {"estado": True}

def visualizar_todo(video_path,muestraid,videoid):
    cap = cv2.VideoCapture(video_path)
    contador = 0
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


    #cruce de rodillas 
    cruce_rodillas_indicador = False
    orientacion=1 #por defecto se inicia frontal
    #orientacion 1= frontal, 2= espalda, 3= lateral

    #reproducir_flag = False
    #pausado_flag = False

    while cap.isOpened():
        #if reproducir:
            #reproducir_flag["estado"] = True
            #pausado_flag = False
        #if pausar:
            #pausado_flag = True
            #reproducir_flag = False


        if reproducir_flag["estado"]:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (config.ANCHO, config.ALTO))

            #aplicacion de mejoras del frame 
            frame_mejorado, scale  = config.mejorar_frame(frame)
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
                    x_s, y_s = config.suavizar_landmark(idx, x_orig, y_orig)

                    xs.append(x_s)
                    ys.append(y_s)

                min_x, max_x = int(min(xs)), int(max(xs))
                min_y, max_y = int(min(ys)), int(max(ys))

                padding_x = int((max_x - min_x) * config.padding)
                padding_y = int((max_y - min_y) * config.padding)

                min_x = max(0, min_x - padding_x)
                max_x = min(w, max_x + padding_x)
                min_y = max(0, min_y - padding_y)
                max_y = min(h, max_y + padding_y)

                # Dibujar recuadro
                cv2.rectangle(frame_mejorado, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
                # Display the frame using Streamlit's st.image
                #frame_placeholder.image(frame_mejorado, channels="BGR")

                # Normalización
                persona_recortada = frame[min_y:max_y, min_x:max_x]
                persona_normalizada = cv2.resize(persona_recortada, (config.NORMALIZADO_ANCHO, config.NORMALIZADO_ALTO))
                #fondo_negro = np.zeros_like(persona_normalizada)
                gris_claro = 250
                fondo_negro = np.full_like(persona_normalizada, fill_value=gris_claro)

                nueva_w, nueva_h = config.NORMALIZADO_ANCHO, config.NORMALIZADO_ALTO
                escala_x = nueva_w / (max_x - min_x)
                escala_y = nueva_h / (max_y - min_y)
                
                # Dibujar conexiones suavizadas
                #for connection in mp_pose.POSE_CONNECTIONS:
                    #start_idx, end_idx = connection
                    #if start_idx not in config.puntos_rostro and end_idx not in config.puntos_rostro:
                        #x1 = int((xs[start_idx] - min_x) * escala_x)
                        #y1 = int((ys[start_idx] - min_y) * escala_y)
                        #x2 = int((xs[end_idx] - min_x) * escala_x)
                        #y2 = int((ys[end_idx] - min_y) * escala_y)
                        #cv2.line(fondo_negro, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Dibujar puntos suavizados
                #for idx in range(len(xs)):
                    #if idx not in config.puntos_rostro:
                        #x = int((xs[idx] - min_x) * escala_x)
                        #y = int((ys[idx] - min_y) * escala_y)
                        #cv2.circle(fondo_negro, (x, y), 3, (0, 113, 0), -1)
                        #cv2.putText(fondo_negro, f'p{idx} ({x},{y})', (x + 5, y - 5),
                                #cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                

                # Dibujar conexiones suavizadas
                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx not in config.puntos_rostro and end_idx not in config.puntos_rostro:
                        x1, y1 = int(xs[start_idx]), int(ys[start_idx])
                        x2, y2 = int(xs[end_idx]), int(ys[end_idx])
                        cv2.line(frame_mejorado, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Dibujar puntos suavizados
                for idx, (x, y) in enumerate(zip(xs, ys)):
                    if idx not in config.puntos_rostro:
                        cv2.circle(frame_mejorado, (int(x), int(y)), 3, (0, 255, 0), -1)
                
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

                #orientacion = "Lateral" if (cruce_caderas or cruce_hombros or cruce_rodillas or cruce_rodillas_2 or
                                        #(diff_caderas < 20 and diff_hombros < 20)) else "Frente o Espalda"

                # Mostrar orientación sobre el frame
                #cv2.putText(frame_mejorado, f'Orientacion: {orientacion}', (10, 20),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


                #-------------------------------------------PUNTOS DISTANCIAS--------------------------------------
                    #32_31
                x1_31 = config.obtener_escala_x(min_x,escala_x,xs[31])
                y1_31 = config.obtener_escala_y(min_y,escala_y,ys[31]) 

                x1_32 = config.obtener_escala_x(min_x,escala_x,xs[32])
                y1_32 = config.obtener_escala_y(min_y,escala_y,ys[32]) 

                distancia_32_21= 0
                distancia_32_21 = config.distancia_euclidiana(x1_32,x1_31,y1_32,y1_31)
                vector_distancia_32_31.append(distancia_32_21)

                    #28_27
                x1_27 = config.obtener_escala_x(min_x,escala_x,xs[27])
                y1_27 = config.obtener_escala_y(min_y,escala_y,ys[27]) 

                x1_28 = config.obtener_escala_x(min_x,escala_x,xs[28])
                y1_28 = config.obtener_escala_y(min_y,escala_y,ys[28]) 

                distancia_28_27= 0
                distancia_28_27 = config.distancia_euclidiana(x1_28,x1_27,y1_28,y1_27)
                vector_distancia_28_27.append(distancia_28_27)

                    #26_25
                x1_25 = config.obtener_escala_x(min_x,escala_x,xs[25])
                y1_25 = config.obtener_escala_y(min_y,escala_y,ys[25])

                x1_26 = config.obtener_escala_x(min_x,escala_x,xs[26])
                y1_26 = config.obtener_escala_y(min_y,escala_y,ys[26])

                distancia_26_25 =0
                distancia_26_25 =config.distancia_euclidiana(x1_26,x1_25,y1_26,y1_25)
                vector_distancia_26_25.append(distancia_26_25)

                    #31_23
                x1_23 = config.obtener_escala_x(min_x,escala_x,xs[23])
                y1_23 = config.obtener_escala_y(min_y,escala_y,ys[23])

                x1_31 = config.obtener_escala_x(min_x,escala_x,xs[31])
                y1_31 = config.obtener_escala_y(min_y,escala_y,ys[31])

                distancia_31_23 =0
                distancia_31_23 = config.distancia_euclidiana(x1_31,x1_23,y1_31,y1_23)
                vector_distancia_31_23.append(distancia_31_23)

                    #32_24
                x1_24 = config.obtener_escala_x(min_x,escala_x,xs[24])
                y1_24 = config.obtener_escala_y(min_y,escala_y,ys[24])

                x1_32 = config.obtener_escala_x(min_x,escala_x,xs[32])
                y1_32 = config.obtener_escala_y(min_y,escala_y,ys[32])

                distancia_32_24 =0
                distancia_32_24 =config.distancia_euclidiana(x1_32,x1_24,y1_32,y1_24)
                vector_distancia_32_24.append(distancia_32_24)

                    #nuevos puntos 
                    #16_12
                x1_12 = config.obtener_escala_x(min_x,escala_x,xs[12])
                y1_12 = config.obtener_escala_y(min_y,escala_y,ys[12])
                x1_16 = config.obtener_escala_x(min_x,escala_x,xs[16])
                y1_16 = config.obtener_escala_y(min_y,escala_y,ys[16])

                distancia_16_12 = 0
                distancia_16_12 = config.distancia_euclidiana(x1_16,x1_12,y1_16,y1_12)
                vector_distancia_16_12.append(distancia_16_12)

                    #15_11
                x1_11 = config.obtener_escala_x(min_x,escala_x,xs[11])
                y1_11 = config.obtener_escala_y(min_y,escala_y,ys[11])
                x1_15 = config.obtener_escala_x(min_x,escala_x,xs[15])
                y1_15 = config.obtener_escala_y(min_y,escala_y,ys[15])

                distancia_15_11 = 0
                distancia_15_11 = config.distancia_euclidiana(x1_15,x1_11,y1_15,y1_11)
                vector_distancia_15_11.append(distancia_15_11)

                    #32_16
                x1_16 = config.obtener_escala_x(min_x,escala_x,xs[16])
                y1_16 = config.obtener_escala_y(min_y,escala_y,ys[16])
                x1_32 = config.obtener_escala_x(min_x,escala_x,xs[32])
                y1_32 = config.obtener_escala_y(min_y,escala_y,ys[32])
                distancia_32_16 = 0
                distancia_32_16 = config.distancia_euclidiana(x1_32,x1_16,y1_32,y1_16)
                vector_distancia_32_16.append(distancia_32_16)

                    #31_15
                x1_15 = config.obtener_escala_x(min_x,escala_x,xs[15])
                y1_15 = config.obtener_escala_y(min_y,escala_y,ys[15])
                x1_31 = config.obtener_escala_x(min_x,escala_x,xs[31])
                y1_31 = config.obtener_escala_y(min_y,escala_y,ys[31])

                distancia_31_15 = 0
                distancia_31_15 = config.distancia_euclidiana(x1_31,x1_15,y1_31,y1_15)
                vector_distancia_31_15.append(distancia_31_15)

                #--------------------------------------------------------------------------------------------------
                #deteccion de la orientacion de la persona Frontal, Espalda, Lateral
                mano_izquierda= int((xs[16] - min_x) * escala_x)
                mano_derecha = int((xs[15] - min_x) * escala_x)
                #print(f"mano izquierda: {mano_izquierda} - mano derecha: {mano_derecha}")

                #en los primeros 10 fotogramas se evalua la orientacion si esque el cruce de rodillas es false
                if contador<=10 and cruce_rodillas_indicador == False:
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

                if pie_xd >= 560 or pie_xd_2>=560:
                    #print(f"no se realizan calculos")
                    contador=0
                    vector_distancia_32_31.clear()
                    
                    #marcha_fuera.write(f"Marcha fuera de rango")
                else:
                    #marcha_fuera.write(f"")

                    #actualizar las graficas de la caminatas por puntos
                    #df_32_31 = pd.DataFrame({"y": vector_distancia_32_31})
                    #vector_distancia_32_31_placeholder.line_chart(df_32_31)

                    #df_28_27 = pd.DataFrame({"y": vector_distancia_28_27})
                    #vector_distancia_28_27_placeholder.line_chart(df_28_27)

                    #df_26_25 = pd.DataFrame({"y": vector_distancia_26_25})
                    #vector_distancia_26_25_placeholder.line_chart(df_26_25)

                    #reiniciar el contador cuando se considere que hubo una marcha completa de 25s
                    if (contador>=25):

                        #llamar a la funcion de prediccion
                        #vectores = {
                        #"26_25": vector_distancia_26_25,
                        #"28_27": vector_distancia_28_27,
                        #"31_23": vector_distancia_31_23,
                        #"32_24": vector_distancia_32_24,
                        #"32_31": vector_distancia_32_31,
                        #"16_12": vector_distancia_16_12,
                        #"15_11": vector_distancia_15_11,
                        #"32_16": vector_distancia_32_16,
                        #"31_15": vector_distancia_31_15
                        #}
                        #calcular la desviacion y el promedio para poder guardar como muestra    
                        promedio_32_31 = config.obtener_promedio(vector_distancia_32_31)
                        desviacion_32_31 = config.obtener_desviacion(vector_distancia_32_31)
                        promedio_28_27 = config.obtener_promedio(vector_distancia_28_27)
                        desviacion_28_27 = config.obtener_desviacion(vector_distancia_28_27)
                        promedio_26_25 = config.obtener_promedio(vector_distancia_26_25)
                        desviacion_26_25 = config.obtener_desviacion(vector_distancia_26_25)
                        promedio_31_23 = config.obtener_promedio(vector_distancia_31_23)
                        desviacion_31_23 = config.obtener_desviacion(vector_distancia_31_23)
                        promedio_32_24 = config.obtener_promedio(vector_distancia_32_24)
                        desviacion_32_24 = config.obtener_desviacion(vector_distancia_32_24)
                        # NUEVOS PUNTOS PARA REALIZAR MAS PRUEBAS
                        promedio_16_12 = config.obtener_promedio(vector_distancia_16_12)
                        desviacion_16_12 = config.obtener_desviacion(vector_distancia_16_12)
                        promedio_15_11 = config.obtener_promedio(vector_distancia_15_11)
                        desviacion_15_11 = config.obtener_desviacion(vector_distancia_15_11)
                        promedio_32_16 = config.obtener_promedio(vector_distancia_32_16)
                        desviacion_32_16 = config.obtener_desviacion(vector_distancia_32_16)
                        promedio_31_15 = config.obtener_promedio(vector_distancia_31_15)
                        desviacion_31_15 = config.obtener_desviacion(vector_distancia_31_15)
                    

                        #guardar las muestras en la BD
                        #sv.registrar_puntos_muestra(videoid,muestraid,promedio_32_31,desviacion_32_31,promedio_28_27,desviacion_28_27,promedio_26_25,desviacion_26_25,promedio_31_23,desviacion_31_23,promedio_32_24,desviacion_32_24,promedio_16_12,desviacion_16_12,promedio_15_11,desviacion_15_11,promedio_32_16,desviacion_32_16,promedio_31_15,desviacion_31_15,orientacion)

                        #limpiar los vectores
                        vector_distancia_26_25.clear()
                        vector_distancia_28_27.clear()
                        vector_distancia_31_23.clear()
                        vector_distancia_32_24.clear()
                        vector_distancia_32_31.clear()
                        vector_distancia_16_12.clear()
                        vector_distancia_15_11.clear()
                        vector_distancia_32_16.clear()
                        vector_distancia_31_15.clear()

                        cruce_rodillas_indicador= False
                        contador=0

                # Incrementar contador
                contador += 1
                 
                #contador_stream.write(f"Fotograma: {contador}")
                orientacionString = "Frontal" if orientacion == 1 else "Espalda" if orientacion == 2 else "Lateral"
                print(orientacionString)
                #orientacion_stream.write(f"Orientacion: {orientacionString}")
                cv2.line(fondo_negro, (nueva_w // 2, 0), (nueva_w // 2, nueva_h), (255, 255, 0), 2)
                cv2.line(fondo_negro, (0, nueva_h // 2), (nueva_w, nueva_h // 2), (255, 0, 0), 2)

                #linea limite para los pies, si se pasa de esta linea entonces no tomar datos
                y_linea = int(nueva_h * 0.96)
                cv2.line(fondo_negro, (0, y_linea), (nueva_w, y_linea), (255, 0, 0), 2)
                
                #normalizacion.image(fondo_negro, channels="BGR")

                # Mostrar imagen final directamente
                cv2.imshow('Visualizacion Completa', frame_mejorado)                

            else:
                cv2.imshow('Visualizacion Completa', frame_mejorado)

            if cv2.waitKey(config.delay) & 0xFF == ord('q'):
                break
        #else:
            #time.sleep(0.1)  # espera mientras está pausado

    cap.release()
    cv2.destroyAllWindows()


#"""
if __name__ == "__main__":
    video_ruta="Nuevos/VelezV/Controlado/Frontal/1.mp4"
    visualizar_todo(video_ruta,1,1)
#"""


"""

if __name__ == "__main__":

    escenarios = ["Controlado", "NoControlado"]
    participantes = ["MendozaA"]
    Orientacion = ["Frontal", "Espalda", "Lateral"]
    #se obtiene la lista de carpetas de los participantes para que este paso se realize automaticamente
    #participantes=config.obtener_participantes()
   
    participantesID =[]
    for p in participantes:
        #1 primero registrar al participante, si no existe crearlo y devolver el id
        participanteID = sv.regitrarParticipante(p)
        print(f"Participante {p} ID: {participanteID}")
        #2 crear una muestra del participante y devolver el id de la muestra
        muestraid = sv.regitrarMuestra(participanteID)
        print(f"Muestra {muestraid} registrada para el participante {p}")
        
        #recorrer los escenarios
        for escenario in escenarios:
            num = 3
            if escenario == "NoControlado":
                num=1

            for x in Orientacion:
                for j in range(num):
                    print("-------------------------------------------------------------------------------------" )
                    print(f"Participante = {p} Escenario ={escenario} Video ={j+1}" )

                    # primero intenta con mp4
                    ruta_video =""
                    ruta_video_mp4 = f"Nuevos/{p}/{escenario}/{x}/{j+1}.mp4"
                    ruta_video_mov = f"Nuevos/{p}/{escenario}/{x}/{j+1}.mov"

                    if os.path.exists(ruta_video_mp4):
                        ruta_video = ruta_video_mp4
                    elif os.path.exists(ruta_video_mov):
                        ruta_video = ruta_video_mov
                    else:
                        print(f"⚠ No se encontró el video {j+1} en formato mp4 ni mov.")
                        continue

                    #3 registrar el video y devolver el id del video para que se guarden los datos en la funcion siguiente
                    videoID = sv.registrarVideo(muestraid)
                    print(f"Video {videoID} registrado para la muestra {muestraid} del participante {p}")
                    visualizar_todo(ruta_video,muestraid,videoID)


    #print (f"PROCESO FINALIZADO REVISAR LAS CONSULTAS DE LA BD CON evaluacionID -> {evaluacionID}")
    print(f"OBTENCION DE MUESTRAS FINALIZADO, ENTRENAR LOS MODELOS ES EL SIGUIENTE PASO")



"""