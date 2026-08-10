import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from groq import Groq

# Configuración de la página para que sea amplia y profesional
st.set_page_config(
    page_title="Detección de Enfermedades en Hojas de Café",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clases basadas en las que detectó tu modelo (ajusta los nombres si es necesario)
CLASES = ['Sana / Sin síntomas', 'Roya (Hemileia vastatrix)', 'Cercospora (Mancha de Hielo)', 'Minador de la hoja']

# Cargar el modelo entrenado
@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model('modelo_hojas_cafe.h5')
    return modelo

try:
    model = cargar_modelo()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# Función para consultar a la API de Groq y obtener recomendaciones técnicas
def obtener_recomendaciones_groq(enfermedad, confianza):
    # Inicializa el cliente de Groq (asegúrate de configurar tu API Key en los secretos de Streamlit o variable de entorno)
    api_key = st.secrets.get("GROQ_API_KEY", "TU_API_KEY_AQUI")
    client = Groq(api_key=api_key)
    
    prompt = f"""
    Actúa como un experto agrónomo del IHCAFE. Se ha detectado la enfermedad '{enfermedad}' en una hoja de café con una confianza del {confianza:.1f}%.
    Proporciona una respuesta estructurada exactamente con estas 5 secciones técnicas:
    1. Diferenciación a simple vista.
    2. Manejo agronómico preventivo y correctivo.
    3. Consulta a un técnico IHCAFE.
    4. Monitoreo y seguimiento.
    5. Registro y trazabilidad.
    Sé conciso, profesional y directo en cada punto.
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"No se pudo conectar con la API de Groq: {e}"

# --- DISEÑO DE LA INTERFAZ (2 Columnas) ---
st.title("Captura de Imagen Foliar")
st.write("Posicione la hoja de café bajo luz natural. El sistema detectará automáticamente signos de roya, cercospora o plagas.")

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("### Opciones de Carga")
    tipo_carga = st.radio("Seleccione método:", ["Subir archivo", "Usar cámara"], horizontal=True)
    
    imagen_cargada = None
    if tipo_carga == "Subir archivo":
        imagen_cargada = st.file_uploader("Cargue una imagen de la hoja", type=["jpg", "jpeg", "png"])
    else:
        imagen_cargada = st.camera_input("Tome una foto")

    if imagen_cargada is not None:
        imagen = Image.open(imagen_cargada)
        st.image(imagen, caption="Imagen analizada", use_column_width=True)

with col2:
    if imagen_cargada is not None:
        with st.spinner("Analizando imagen con Inteligencia Artificial..."):
            # Procesar imagen para el modelo
            img = imagen.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predicción
            prediccion = model.predict(img_array)
            clase_idx = np.argmax(prediccion[0])
            confianza = float(np.max(prediccion[0]) * 100)
            
            # Ajustar índice de clase si está dentro de los límites
            enfermedad_detectada = CLASES[clase_idx] if clase_idx < len(CLASES) else "Desconocido"

        # Encabezado de Resultados
        st.markdown(f"### {enfermedad_detectada}")
        st.markdown(f"**Confianza:** `{confianza:.1f}%`")
        
        st.markdown("---")
        st.markdown("#### 🌿 ORIENTACIÓN Y MANEJO PREVENTIVO")
        st.write("Aquí tienes una recomendación técnica detallada para manejar la situación:")
        
        # Obtener recomendaciones de Groq
        with st.spinner("Generando recomendaciones técnicas con Groq..."):
            recomendaciones = obtener_recomendaciones_groq(enfermedad_detectada, confianza)
            st.markdown(recomendaciones)
    else:
        st.info("👈 Por favor, suba o capture una imagen en el panel izquierdo para ver el diagnóstico y las recomendaciones.")
