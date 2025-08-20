from db import get_connection

def ObtenerDatosEntrenamiento():
    try:
      # Paso 1: Obtener JSON desde PostgreSQL
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT obtener_datos_entrenamiento(%s, %s)", (1, 1))
        resultado = cur.fetchone()[0]  # El JSON como dict/list
        cur.close()
        conn.close()
        return resultado
    except Exception as e:
        print(f"Error al cargar los datos de entrenamiento: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()