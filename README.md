# TinyLLaMA + Traducción (Flask)

Aplicación Flask que:
1) Detecta si el prompt está en español.
2) Traduce ES→EN con modelos Marian.
3) Llama a TinyLLaMA local vía `llama_cpp` (GGUF en CPU).
4) Traduce la respuesta EN→ES y la muestra en un frontend muy simple.

> Este repo **no incluye** los pesos de los modelos por tamaño. En la carpeta `models/` hay instrucciones para colocarlos localmente.

---

## Requisitos

- Python 3.10+ (Windows o Linux)
- CPU (sin GPU)
- Modelos Marian EN↔ES y un archivo GGUF de TinyLLaMA (ver `models/README.md`).

### Dependencias Python


>> pip install -r requirements.txt


Nota Torch (Linux/CPU): si pip install torch falla o descarga un wheel incorrecto, instala el wheel CPU explícito:
>> pip install --index-url https://download.pytorch.org/whl/cpu torch
