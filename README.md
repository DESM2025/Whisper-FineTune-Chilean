Proyecto de Deep Learning que adapta el modelo Whisper Tiny de OpenAI para mejorar el reconocimiento de voz con acento chileno

### Autores
* Diego Silva
* Pablo Ibañez

**Profesor:** Jaime Jimenez Ruiz
**Curso:** Deep Learning segundo semestre 2025

### Descripcion del Proyecto

Este proyecto implementa fine-tuning del modelo Whisper Tiny usando el dataset Google Chilean Spanish (4374 audios) para mejorar la transcripcion automatica de voz con acento chileno,el modelo se entrena durante 4 epocas con division 60/20/20 (train/validation/test) y alcanza buenos resultados en la adaptacion al dialecto regional.

### Caracteristicas 

* Preprocesamiento de audio a 16kHz
* Normalizacion de texto (minusculas, sin puntuacion)
* Entrenamiento con precision mixta FP16
* Evaluacion con metrica WER (Word Error Rate)
* Visualizaciones de curvas de aprendizaje
* Mapas de atencion XAI para interpretabilidad
* Interfaz web interactiva con Streamlit

### Requisitos

* Python 3.10
* CUDA 12.1+ (GPU con 6GB+ VRAM recomendado)
* Conda o Miniconda
* El entrenamiento completo toma aproximadamente 25 minutos en una RTX 4060

### Crear entorno conda
conda env create -f environment.yml
conda activate whisper-finetune


### Ejecutar Demo Web

* streamlit run app.py

## Estructura del Proyecto

```
.
├── Audio_a_texto.ipynb           # Notebook principal de entrenamiento
├── app.py                         # Interfaz web Streamlit
├── environment.yml                # Dependencias del proyecto
├── README.md                      # Este archivo
├── whisper_tiny_chileno_final/   # Modelo entrenado (no incluido en repo)
└── whisper-tiny-chilean/         # Checkpoints intermedios (no incluido en repo)
```

### Resultados

El modelo entrenado muestra mejoras en la transcripcion de audio con acento chileno comparado con Whisper Tiny base. Las metricas finales y curvas de aprendizaje se encuentran en el notebook.

### Dataset

- Fuente: [Google Chilean Spanish](https://huggingface.co/datasets/ylacombe/google-chilean-spanish)
- total audios: 4374
- Division: 60% train, 20% validation, 20% test
- Voces: Masculinas y femeninas

### Notas Importantes

1. Los modelos entrenados NO estan incluidos en el repositorio por su tamaño (~150MB)
2. Debes entrenar localmente usando el notebook
3. Si encuentras errores de memoria, reduce `BATCH_SIZE` en el notebook
4. Para audios >30s, el modelo usa chunking automatico en la app

## Tecnologias Utilizadas

- PyTorch 2.5
- Transformers (Hugging Face)
- Whisper (OpenAI)
- Streamlit
- Librosa
- CUDA 12.1
