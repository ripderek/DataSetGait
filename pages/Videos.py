#en este archivo listar los videos para poderlos ver en la app web
#tambien permitir seleccionar un video para hacer la prueba de como se toman las muestras
#tambien permitir seleccionar un video para hacer la prueba de como se predice con los modelos ya creados

import streamlit as st
import os
import urllib.parse

BASE_PATH = "Participantes"

def mostrar_treeview(ruta):
    elementos = sorted(os.listdir(ruta))
    for elemento in elementos:
        full_path = os.path.join(ruta, elemento)
        if os.path.isdir(full_path):
            with st.expander("📁 " + elemento, expanded=False):
                mostrar_treeview(full_path)
        else:
            if elemento.lower().endswith((".mp4", ".mov")):
                # Convertir ruta a URL-safe
                video_param = urllib.parse.quote(full_path)

                # Generar URLs
                url_identificacion = f"./VerIdentificacion?video={video_param}"
                url_muestra = f"./VerMuestra?video={video_param}"

                # Mostrar nombre y botones
                st.write(f"🎥 {elemento}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"[🔎 Ver Identificación]({url_identificacion})", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"[🧪 Ver Muestra]({url_muestra})", unsafe_allow_html=True)

            else:
                st.write("📄", elemento)

def main():
    st.set_page_config(page_title="TreeView de Participantes", layout="wide")
    st.title("📂 Explorador de Participantes")
    mostrar_treeview(BASE_PATH)

if __name__ == "__main__":
    main()
