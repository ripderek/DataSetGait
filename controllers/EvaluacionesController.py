from db import get_connection

def CrearGuardarNuevaEv(nombreEv):
    try:
        conn = get_connection()
        cursor = conn.cursor()
          # OJO: hay que pasar la tupla (nombreEv,) con la coma
        cursor.execute("SELECT insertar_evaluacion_retornar_id(%s)", (nombreEv,))
        conn.commit()
            # fetchone devuelve la primera fila
        fila = cursor.fetchone()
        cursor.close()
        conn.close()
        # fila es una tupla, en tu caso con un solo valor
        evaluacionid = fila[0] if fila else None
        return evaluacionid
    except Exception as e:
        print(f"Error al guardar CrearGuardarNuevaEven la bd: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def RegistrarParticipanteEv(participante, evaluacionID):
    try:
        conn = get_connection()
        cursor = conn.cursor()
          # OJO: hay que pasar la tupla (nombreEv,) con la coma
        cursor.execute("SELECT insertar_participante_evaluacion_retornar_id(%s, %s)", (participante, evaluacionID))
        conn.commit()
        # fetchone devuelve la primera fila
        fila = cursor.fetchone()
        cursor.close()
        conn.close()
        # fila es una tupla, en tu caso con un solo valor
        evaluacionpid = fila[0] if fila else None
        return evaluacionpid
    except Exception as e:
        print(f"Error al guardar RegistrarParticipanteEv en la bd: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def InsertarResultadosVideos( n_video_p,orientacion_p,escenario_p,vp_p,fp_p,pi_p,pi_vp_p,pc_p,pc_i_p,evaluacionpid_p,evaluacionID):
    try:
        conn = get_connection()
        cursor = conn.cursor()
          # OJO: hay que pasar la tupla (nombreEv,) con la coma %s
        cursor.execute("SELECT insertar_resultados_videos(%s, %s, %s,%s,%s,%s,%s,%s,%s,%s,%s)", (	
            n_video_p,
	        orientacion_p,
	        escenario_p,
	        vp_p,
	        fp_p,
	        pi_p,
	        pi_vp_p,
	        pc_p,
	        pc_i_p,
	        evaluacionpid_p,evaluacionID))
        conn.commit()
        # fetchone devuelve la primera fila
        fila = cursor.fetchone()
        cursor.close()
        conn.close()
        # fila es una tupla, en tu caso con un solo valor
        evaluacionpid = fila[0] if fila else None
        return evaluacionpid
    except Exception as e:
        print(f"Error al guardar InsertarResultadosVideos en la bd: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()