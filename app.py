import streamlit as st
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

st.set_page_config(
    page_title="Whisper Chileno - Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title {text-align: center; font-size: 3em; font-weight: bold; color: #2E86C1; margin-bottom: 0px;}
    .subtitle {text-align: center; font-size: 1.2em; color: #566573; margin-bottom: 30px;}
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Proyecto Semestral")
    st.markdown("---")
    st.metric(label="Fecha", value=datetime.now().strftime("%d/%m/%Y"))
    st.info("**Modelo:** Whisper Tiny (Fine-Tuned)\n\n**Dataset:** Spanish-Chilean\n\n**GPU:** RTX 4060")
    st.caption("Deep Learning 2025 diego silva/pablo ibañes")

# CARGA DEL MODELO 
@st.cache_resource
def load_model_pipeline():
    MODEL_PATH = "whisper_tiny_chileno_final"
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: No se encuentra la ruta: {MODEL_PATH}")
        return None, None
    try:
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH, attn_implementation="eager")
        return processor, model
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None, None

processor, model = load_model_pipeline()

# FUNCIONES
def process_audio(audio_file):
    # Detectar extension para evitar errores
    file_extension = audio_file.name.split(".")[-1].lower()
    temp_filename = f"temp_audio_input.{file_extension}"
    
    with open(temp_filename, "wb") as f:
        f.write(audio_file.getbuffer())
    
    try:
        # Cargar audio a 16kHz
        speech_array, sampling_rate = librosa.load(temp_filename, sr=16000)
        return speech_array, sampling_rate, temp_filename
    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise e

def plot_attention(model, processor, inputs, predicted_ids):
    outputs = model(input_features=inputs, decoder_input_ids=predicted_ids, output_attentions=True)
    cross_attentions = outputs.cross_attentions[-1]
    attention_matrix = cross_attentions[0].mean(dim=0).detach().cpu().numpy()
    
    if attention_matrix.shape[0] != len(predicted_ids[0]):
        attention_matrix = attention_matrix.T
    
    tokens = processor.tokenizer.convert_ids_to_tokens(predicted_ids[0])
    cleaned_tokens = [t.replace('Ġ', ' ').replace('Ċ', '') for t in tokens]
    
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(attention_matrix, xticklabels=False, yticklabels=cleaned_tokens, cmap="plasma", cbar_kws={'label': 'Atención'}, ax=ax)
    ax.set_title("XAI: Sincronización Audio-Texto (Primeros 30s)", fontsize=14, color="white")
    ax.set_xlabel("Tiempo (Audio)", fontsize=12)
    ax.set_ylabel("Tokens", fontsize=12)
    return fig

st.markdown('<div class="main-title">🇨🇱 Inferencia de Voz</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Modelo adaptado al acento chileno</div>', unsafe_allow_html=True)

col_izq, col_der = st.columns([1, 1.5], gap="large")

# COLUMNA IZQUIERDA
with col_izq:
    with st.container(border=True):
        st.subheader("1. Entrada de Audio")
        uploaded_file = st.file_uploader("Sube un archivo (MP3/WAV)", type=["mp3", "wav", "m4a"])
        
        if uploaded_file is not None:
            st.audio(uploaded_file)
            st.markdown("---")
            
            if st.button("Transcribir Audio", type="primary", use_container_width=True):
                if model is not None:
                    with st.spinner("Procesando audio completo..."):
                        temp_file = None
                        try:
                            # Cargar Audio
                            speech_array, sampling_rate, temp_file = process_audio(uploaded_file)
                            
                            # Detectar dispositivo (GPU o CPU)
                            device = "cuda:0" if torch.cuda.is_available() else "cpu"
                            
                            # Crear pipeline automático
                            pipe = pipeline(
                                "automatic-speech-recognition",
                                model=model,
                                tokenizer=processor.tokenizer,
                                feature_extractor=processor.feature_extractor,
                                chunk_length_s=30, 
                                device=device
                            )
                            
                            # Generar texto completo forzando español
                            result = pipe(
                                temp_file, 
                                generate_kwargs={"language": "spanish", "task": "transcribe"}
                            )
                            transcription_full = result["text"]

                            # Seguimos usando el metodo manual solo para obtener la matriz de atención
                            inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt").input_features
                            # Recortamos si es muy largo solo para el gráfico (evita errores de memoria)
                            if inputs.shape[-1] > 3000: 
                                inputs = inputs[:, :, :3000]
                            
                            if torch.cuda.is_available():
                                inputs = inputs.to("cuda")
                                
                            predicted_ids_xai = model.generate(
                                inputs, 
                                num_beams=5, 
                                language="spanish", 
                                task="transcribe"
                            )
                            
                            # Guardar en sesion
                            st.session_state['transcription'] = transcription_full
                            st.session_state['fig'] = plot_attention(model, processor, inputs.to("cpu"), predicted_ids_xai.to("cpu"))
                        
                        except Exception as e:
                            st.error(f"Error: {e}")
                        finally:
                            if temp_file and os.path.exists(temp_file):
                                os.remove(temp_file)

# COLUMNA DERECHA
with col_der:
    with st.container(border=True):
        st.subheader("2. Resultados")
        
        if 'transcription' in st.session_state:
            st.success("Inferencia Completada")
            
            st.markdown("#### Texto Transcrito")
            # Usamos un área de texto con scroll para audios largos
            st.text_area("", value=st.session_state['transcription'], height=200)
            
            st.markdown("#### Explicabilidad (XAI - Inicio)")
            st.pyplot(st.session_state['fig'])
            
            if st.button("Limpiar Resultados"):
                del st.session_state['transcription']
                st.rerun()
        else:
            st.info("Esperando entrada...")
            st.markdown(
                """<div style="text-align: center; color: gray; padding: 50px;">
                <h3>Resultados aparecerán aquí</h3></div>""", 
                unsafe_allow_html=True
            )