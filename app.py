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

# Clases de tu modelo (asegúrate de que coincidan con el orden alfabético de tu entrenamiento en Colab)
CLASES = ['Cercospora (Mancha de Hierro)', 'Plagas / Minador', 'Roya (Hemileia vastatrix)', 'Sana / Sin síntomas']

# Cargar el modelo ligero TFLite directamente desde GitHub
@st.cache_resource
def cargar_modelo_tflite():
    ruta_modelo = "modelo_hojas_cafe.tflite"
    if os.path.exists(ruta_modelo):
        interpreter = tf.lite.Interpreter(model_path=ruta_modelo)
        interpreter.allocate_tensors()
        return interpreter
    else:
        return None

interpreter = cargar_modelo_tflite()

# Estructura principal en dos columnas
col_izq, col_der = st.columns([1.1, 1.9], gap="large")

with col_izq:
    st.markdown("### Captura de Imagen Foliar")
    st.caption("Posicione la hoja de café bajo luz natural. El sistema detectará automáticamente signos de roya, cercospora o plagas.")
    
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
    
    if imagen_cargada is not None and interpreter is not None:
        # Preprocesamiento de la imagen
        img_resized = imagen.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predicción usando TFLite Interpreter
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediccion = interpreter.get_tensor(output_details[0]['index'])[0]
        
        clase_idx = np.argmax(prediccion)
        confianza = float(prediccion[clase_idx] * 100)
        enfermedad = CLASES[clase_idx] if clase_idx < len(CLASES) else "Desconocido"
        
        # Umbral de seguridad para mitigar falsos positivos en hojas sanas
        UMBRAL_CONFIANZA = 75.0

        res_col1, res_col2 = st.columns([3, 1])
        with res_col1:
            if confianza < UMBRAL_CONFIANZA:
                st.markdown(f"⚠️ **Posible falso positivo / Incierto**")
                st.caption(f"El modelo apunta a {enfermedad}, pero la confianza es baja ({confianza:.1f}%). Podría tratarse de una hoja sana.")
            else:
                st.markdown(f"## {enfermedad}")
                st.caption("Diagnóstico procesado por red neuronal optimizada (TFLite).")
        with res_col2:
            color_conf = "#d32f2f" if confianza < UMBRAL_CONFIANZA else "#2e7d32"
            st.markdown(f"<h1 style='text-align: right; color: {color_conf};'>{confianza:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: right; color: gray; font-size: 10px;'>CONFIANZA</p>", unsafe_allow_html=True)
        
        # Desglose detallado de probabilidades para depurar si el modelo duda
        with st.expander("📊 Ver desglose detallado de probabilidades por clase"):
            for i, prob in enumerate(prediccion):
                st.progress(float(prob), text=f"{CLASES[i]}: {prob*100:.2f}%")
        
        st.markdown("---")
        st.markdown("💡 **ORIENTACIÓN Y MANEJO PREVENTIVO**")
        st.markdown("Aquí tienes una recomendación técnica detallada para manejar la situación:")
        
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
            <b>03. Consulta a un técnico especializado</b><br>
            <span style="font-size: 14px; color: #4a4a4a;">Consulte si las afecciones superan el 30% del follaje total. Un técnico evaluará niveles nutricionales en suelo para descartar problemas primarios.</span>
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
        
        if interpreter is None:
            st.warning("⚠️ No se encontró el archivo `modelo_hojas_cafe.tflite` en el repositorio de GitHub. Sube tu archivo `.tflite` para continuar.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 11px;'>© 2026 AGRODETECT - SOPORTE TÉCNICO</p>", unsafe_allow_html=True)
