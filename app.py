import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuración de la página en modo ancho
st.set_page_config(
    page_title="AgroDetect - Detección de Enfermedades en Hojas de Café",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clases de tu modelo (asegúrate de que coincidan con el orden de tus carpetas del dataset)
CLASES = ['Sana / Sin síntomas', 'Roya (Hemileia vastatrix)', 'Cercospora (Mancha de Hierro)', 'Plagas / Minador']

# Función para cargar el modelo guardado
@st.cache_resource
def cargar_modelo():
    ruta_modelo = "modelo_hojas_cafe.h5"
    if os.path.exists(ruta_modelo):
        modelo = tf.keras.models.load_model(ruta_modelo)
        return modelo
    else:
        return None

modelo = cargar_modelo()

# Estructura principal en dos columnas (igual al diseño de referencia)
col_izq, col_der = st.columns([1.1, 1.9], gap="large")

with col_izq:
    st.markdown("### Captura de Imagen Foliar")
    st.caption("Posicione la hoja de café bajo luz natural. El sistema detectará automáticamente signos de roya, cercospora o plagas.")
    
    # Opciones de carga
    metodo = st.radio("Seleccione método:", ["Subir archivo", "Usar cámara"], horizontal=True)
    
    imagen_cargada = None
    if metodo == "Subir archivo":
        imagen_cargada = st.file_uploader("Cargue una imagen de la hoja", type=["jpg", "jpeg", "png"])
    else:
        imagen_cargada = st.camera_input("Tome una foto de la hoja")

    if imagen_cargada is not None:
        imagen = Image.open(imagen_cargada)
        st.image(imagen, caption="Imagen analizada", use_container_width=True)

with col_der:
    st.markdown("<p style='text-align: right; color: gray; font-size: 12px;'>ÚLTIMO DIAGNÓSTICO</p>", unsafe_allow_html=True)
    
    if imagen_cargada is not None and modelo is not None:
        # Preprocesamiento de la imagen para la CNN
        img_resized = imagen.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predicción
        prediccion = modelo.predict(img_array)
        clase_idx = np.argmax(prediccion[0])
        confianza = float(np.max(prediccion[0]) * 100)
        enfermedad = CLASES[clase_idx] if clase_idx < len(CLASES) else "Desconocido"
        
        # Cabecera de resultados con porcentaje grande a la derecha
        res_col1, res_col2 = st.columns([3, 1])
        with res_col1:
            st.markdown(f"## {enfermedad}")
            st.caption("Diagnóstico procesado por red neuronal convolucional.")
        with res_col2:
            st.markdown(f"<h1 style='text-align: right; color: #2e7d32;'>{confianza:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: right; color: gray; font-size: 10px;'>CONFIANZA</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Bloque de orientación y manejo preventivo (Tarjetas numeradas)
        st.markdown("💡 **ORIENTACIÓN Y MANEJO PREVENTIVO**")
        st.markdown("Aquí tienes una recomendación técnica detallada para manejar la situación:")
        
        # Tarjetas de recomendaciones basadas en el diagnóstico
        st.markdown("""
        <div style="background-color: #fcfaf7; padding: 15px; border-radius: 8px; border: 1px solid #e0dcd0; margin-bottom: 10px;">
            <b>01. Diferenciación a simple vista</b><br>
            <span style="font-size: 14px; color: #4a4a4a;">Observe los patrones de manchas en el haz y envés de la hoja. Valore si presenta halos cloróticos o anillos concéntricos característicos de la lesión detectada.</span>
        </div>
        <div style="background-color: #fcfaf7; padding: 15px; border-radius: 8px; border: 1px solid #e0dcd0; margin-bottom: 10px;">
            <b>02. Manejo agronómico preventivo y correctivo</b><br>
            <span style="font-size: 14px; color: #4a4a4a;">Regular sombra al 40-50% para reducir estrés hídrico. Aplicar caldos minerales preventivos antes de temporadas de alta humedad y evitar exceso de follaje húmedo.</span>
        </div>
        <div style="background-color: #fcfaf7; padding: 15px; border-radius: 8px; border: 1px solid #e0dcd0; margin-bottom: 10px;">
            <b>03. Consulta a un técnico IHCAFE</b><br>
            <span style="font-size: 14px; color: #4a4a4a;">Consulte si las afecciones superan el 30% del follaje total. Un técnico evaluará niveles de Nitrógeno y Potasio en suelo para descartar problemas nutricionales primarios.</span>
        </div>
        <div style="background-color: #fcfaf7; padding: 15px; border-radius: 8px; border: 1px solid #e0dcd0; margin-bottom: 10px;">
            <b>04. Monitoreo y seguimiento</b><br>
            <span style="font-size: 14px; color: #4a4a4a;">Monitoree quincenalmente en épocas críticas. Revise hojas del tercio medio de la planta para detectar brotes tempranos de propagación.</span>
        </div>
        <div style="background-color: #fcfaf7; padding: 15px; border-radius: 8px; border: 1px solid #e0dcd0; margin-bottom: 10px;">
            <b>05. Registro y trazabilidad</b><br>
            <span style="font-size: 14px; color: #4a4a4a;">Documente las condiciones climáticas previas y mantenga un registro histórico por lote para optimizar el manejo integral del cultivo.</span>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.info("👈 Por favor, sube o capture una imagen en el panel izquierdo para ver el diagnóstico y las recomendaciones.")
        
        if modelo is None:
            st.warning("⚠️ No se encontró el archivo `modelo_hojas_cafe.h5` en el repositorio. Asegúrate de haberlo subido o configurado correctamente.")

# Pie de página elegante
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 11px;'>© 2026 AGRODETECT - SOPORTE TÉCNICO</p>", unsafe_allow_html=True)
