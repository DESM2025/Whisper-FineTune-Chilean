Este proyecto implementa el Fine-Tuning del modelo Whisper(Tiny) de OpenAI para mejorar el reconocimiento automático del habla (ASR) en español con acento chileno

Además, incluye una interfaz web interactiva con Streamlit para probar el modelo en tiempo real 
Debido a las restricciones de tamaño de archivo de GitHub el modelo entrenado (`whisper_tiny_chileno_final`) no se incluye

- Entrenamiento: Scripts para re-entrenar Whisper usando Hugging Face Transformers.
- Preprocesamiento:*Pipeline de audio usando `librosa` y `datasets`.
- Demo Interactiva: App web (`app.py`) para transcribir audios y visualizar el espectrograma de Mel.
- Metricas: Evaluación del modelo mediante WER (Word Error Rate).

conda env create -f environment.yml
conda activate whisper-finetune

streamlit run app.py