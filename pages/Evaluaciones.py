#streamlit run prediccion.py
import streamlit as st
import os
import cv2
import mediapipe as mp
import config
import pandas as pd
import Styles.estilos as estilos
import controllers.EvaluacionesController as sv

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils



#estado para contar los VP -> Verdaderos Positivos
VP = {"contador": 0}
#estado para contar los FP -> Falso
# Positivos
FP = {"contador": 0}
#estado para contrar los PI -> precision_identificacion
PI = {"contador": 0}


reproducir_flag = {"estado": False}
#STREAMLIT
# Set the title for the Streamlit app
st.set_page_config(
    page_title="Pruebas",
    layout="wide",  # <--- esto hace que el contenedor ocupe todo el ancho
    initial_sidebar_state="auto"
)
# Título de la app
st.title("Evaluación de Modelos")

# Selección de modelo

# Crear dos columnas: izquierda y derecha
col_izq, col_der = st.columns(2)

# ----- CONTENIDO IZQUIERDO -----
with col_izq:
   
    identificacion_stream = st.empty()  # Esto va al lado izquierdo
    identificacion_stream.markdown(
                                estilos.subtitulo_centrado(f"Identificación ->"),
                                unsafe_allow_html=True
                            )
    frame_placeholder = st.empty()      # Aquí se muestra el video
    reproducir = st.button("Iniciar evaluaciones")
    # pausar = st.button("Pausar")
    
    #3 columnas
    col_izq_1, col_med_1,col_der_1 = st.columns(3)
    with col_izq_1:
        contador_stream = st.empty()
    with col_med_1:
        orientacion_stream = st.empty()
    with col_der_1:
        marcha_fuera = st.empty()
    modelo_seleccionado = st.selectbox("Seleccione un modelo", ["RF", "MLP"])



# ----- CONTENIDO DERECHO -----
with col_der:
    dic_prob_stream = st.empty()
    dic_prob_stream.markdown(
                             estilos.subtitulo_centrado(f"Probabilidades:"),
                                unsafe_allow_html=True
                            )
    diccionario_stream = st.empty()     # Esta tabla va al lado derech



def visualizar_todo(video_path,participante):
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

    persona_identificada = "No identificada"
    persona_identificada2 = "No identificada"

    #cruce de rodillas 
    cruce_rodillas_indicador = False
    orientacion=1 #por defecto se inicia frontal
    #orientacion 1= frontal, 2= espalda, 3= lateral

    #reproducir_flag = False
    #pausado_flag = False

    while cap.isOpened():
        #if reproducir:
            #reproducir_flag["estado"] =True
            #reproducir_flag = True
            #pausado_flag = False
        #if pausar:
            #pausado_flag = True
            #reproducir_flag = False


        #if reproducir_flag["estado"]:
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

                # Normalización (solo para cálculos)
                #persona_recortada = frame[min_y:max_y, min_x:max_x]
                #persona_normalizada = cv2.resize(persona_recortada, (NORMALIZADO_ANCHO, NORMALIZADO_ALTO))

                nueva_w, nueva_h = config.NORMALIZADO_ANCHO, config.NORMALIZADO_ALTO
                escala_x = nueva_w / (max_x - min_x)
                escala_y = nueva_h / (max_y - min_y)

                
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
                    #persona_identificada = "No identificada"
                    marcha_fuera.write(f"Marcha fuera de rango")
                else:
                    tamano_texto = 0.5
                    marcha_fuera.write(f"")
                    cv2.putText(frame_mejorado, f'{persona_identificada}', (int(xs[0]-40), int(ys[0])-30),
                            cv2.FONT_HERSHEY_SIMPLEX, tamano_texto, (255, 255, 255), 2)

                    #reiniciar el contador cuando se considere que hubo una marcha completa de 25s
                    if (contador>=25):
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


                        prediccion,probabilidad_predicha, probabilidades = config.RealizarPrediccion(vectores,orientacion,"RF")#modelo_seleccionado
                        #diccionario_stream.write(f"{probabilidades}")
                        df = pd.DataFrame(list(probabilidades.items()), columns=["Nombre", "Valor"])
                        diccionario_stream.table(df)
                        #print(f"prediccion=>{prediccion}")
                        #si la probabilidad es igual o mas alta que la precision de probabilidad entonces alli si afirmar a la persona identificada

                        if probabilidad_predicha >= config.precision_identificacion: 
                            persona_identificada = f"{prediccion}"
                            #identificacion_stream.write(f"Identificación -> {persona_identificada}   {probabilidad_predicha:.2f} %")
                            identificacion_stream.markdown(
                                estilos.subtitulo_centrado(f"Identificación -> {persona_identificada}   {probabilidad_predicha:.2f} %"),
                                unsafe_allow_html=True
                            )
                            persona_identificada2 = prediccion
                            #print(f"Persona identificada > {config.precision_identificacion} >: {prediccion}")
                            #Calcular el PI    
                            #if prediccion == participante:
                                #PI["contador"] += 1
                                #print(f"(PI)=+1")
                    
                        #Calcular el PI    
                        if persona_identificada2 == participante:
                            PI["contador"] += 1
                            print(f"Se mantiene el (PI)=+1 {PI['contador']}")

                        #Calcular los VP, FP    
                        if prediccion == participante:
                            VP["contador"] += 1
                            #PI["contador"] += 1
                            print(f"(VP)=+1 {VP['contador']}")
                            #print(f"(PI)=+1")
                        else:
                            FP["contador"] += 1
                            print(f"(FP)=+1 {FP['contador']}")

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
         
                #STREAMLIT
                # Display the frame using Streamlit's st.image
                frame_placeholder.image(frame_mejorado, channels="BGR") 
                contador_stream.write(f"Fotograma: {contador}")
                orientacionString = "Frontal" if orientacion == 1 else "Espalda" if orientacion == 2 else "Lateral"
                orientacion_stream.write(f"Orientacion: {orientacionString}")

                # Mostrar imagen final directamente
                #cv2.imshow('Visualizacion Completa', frame_mejorado)

            #else:
                #cv2.imshow('Visualizacion Completa', frame_mejorado)

            if cv2.waitKey(config.delay) & 0xFF == ord('q'):
                break
        #else:
            #time.sleep(0.1)  # espera mientras está pausado

    cap.release()
    cv2.destroyAllWindows()



def Reiniciar_indicadores():
    FP["contador"] =0
    VP['contador'] =0
    PI['contador'] =0


if __name__ == "__main__":
    #NoControlado
    #Controlado
    escenario = "NoControlado"
    #1 primero crear la evaluacion en la bd y guardar el id de la evaluacion
    
    #evaluacionID = sv.CrearGuardarNuevaEv("ModeloRFNC_Ev1_7p1")  #IMPORTANTE CAMBIAR EL NOMBRE PARA QUE SE PUEDA GUARDAR LA EVALUACION
    evaluacionID=38

    #2 ahora registrar a los participantes que se van a evaluar en el modelo con el id de la evaluacion
    #participantes = ["JosselynV"]
    #participantes = ["GamarraA","JosselynV"] #para hacer las primeras pruebas automatizadas se las realiza solo con 3 participantes para supervisarlas

    #se obtiene la lista de carpetas de los participantes para que este paso se realize automaticamente
    participantes=config.obtener_participantes()
    Orientacion = ["Frontal", "Espalda", "Lateral"]
    participantesID =[]
    for p in participantes:
        participanteID = sv.RegistrarParticipanteEv(p,evaluacionID)
        participantesID.append(participanteID)
        print (f"evaluacionID -> {evaluacionID}")
        print (f"participanteID -> {participanteID}")
        for x in Orientacion:
            #Reiniciar_indicadores()
            for j in range(3):
                print("-------------------------------------------------------------------------------------" )
                print(f"Participante = {p} Escenario ={escenario} Video ={j+1}" )

                # primero intenta con mp4
                ruta_video =""
                ruta_video_mp4 = f"Participantes/{p}/{escenario}/{x}/{j+1}.mp4"
                ruta_video_mov = f"Participantes/{p}/{escenario}/{x}/{j+1}.mov"

                if os.path.exists(ruta_video_mp4):
                    ruta_video = ruta_video_mp4
                elif os.path.exists(ruta_video_mov):
                    ruta_video = ruta_video_mov
                else:
                    print(f"⚠ No se encontró el video {j+1} en formato mp4 ni mov.")
                    continue

                visualizar_todo(ruta_video,p)
                print (f"participanteID -> {participanteID}")
                config.GuardarResultados(p,escenario,j+1,x,participanteID,evaluacionID,VP['contador'],FP['contador'],PI['contador'])

                FP["contador"] =0
                VP['contador'] =0
                PI['contador'] =0
                
                #Reiniciar_indicadores()

    print (f"PROCESO FINALIZADO REVISAR LAS CONSULTAS DE LA BD CON evaluacionID -> {evaluacionID}")









