from db import get_connection

#funcion para verificar si existe un participante y si no existe lo crea y al final retorna su id de igual forma
def regitrarParticipante(participante):
    try:
        conn = get_connection()
        cursor = conn.cursor()
          # OJO: hay que pasar la tupla (nombreEv,) con la coma
        cursor.execute("SELECT crear_participante_retornarid(%s)", (participante,))
        conn.commit()
        # fetchone devuelve la primera fila
        fila = cursor.fetchone()
        cursor.close()
        conn.close()
        # fila es una tupla, en tu caso con un solo valor
        evaluacionpid = fila[0] if fila else None
        return evaluacionpid
    except Exception as e:
        print(f"Error al registrar el participante en la bd: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def regitrarMuestra(participanteID):
    try:
        conn = get_connection()
        cursor = conn.cursor()
          # OJO: hay que pasar la tupla (nombreEv,) con la coma
        cursor.execute("SELECT crear_muestra_retornarid(%s)", (participanteID,))
        conn.commit()
        # fetchone devuelve la primera fila
        fila = cursor.fetchone()
        cursor.close()
        conn.close()
        # fila es una tupla, en tu caso con un solo valor
        evaluacionpid = fila[0] if fila else None
        return evaluacionpid
    except Exception as e:
        print(f"Error al registrar la muestra en la bd: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()     

def registrarVideo(muestraID):
    try:
        conn = get_connection()
        cursor = conn.cursor()
          # OJO: hay que pasar la tupla (nombreEv,) con la coma
        cursor.execute("SELECT videoid FROM registrar_obtener_id_muestra_video(%s)", (muestraID,))
        videoid_result = cursor.fetchone()
        videoid = videoid_result[0]
        conn.commit()
        cursor.close()
        conn.close()
        return videoid
    except Exception as e:
        print(f"Error al registrar el video en la bd: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()          

def registrar_puntos_muestra(param1, param2, param3, param4, param5, param6,
                              param7, param8, param9, param10, param11, param12,
                              param13, param14, param15, param16, param17, param18, param19, param20,orientacion):
    print("Funcion registrar puntos")
    try:
        print("registrando puntos")
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            CALL registrar_puntos_muestra(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s)
        """, (
            param1, param2, param3, param4, param5, param6,
            param7, param8, param9, param10, param11, param12,
            param13, param14, param15, param16, param17, param18, param19, param20,orientacion
        ))

        conn.commit() 

        cur.close()
        conn.close()
        print("Puntos registrados")

    except Exception as e:
        print(f"Error al registrar los puntos de la muestra en la bd: {e}")