import os
import time
import json
import uuid
import shlex
from pathlib import Path
import subprocess
from flask import Flask, request, jsonify, render_template_string, g
from transformers import MarianMTModel, MarianTokenizer
import traceback
from llama_cpp import Llama
import torch
import logging


# =========================
# CONFIG & LOGGING
# =========================
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("tinyllama-app")

# === RUTAS LOCALES ===
LLAMA_EXE = r"C:\Users\TU_USUARIO\llama.cpp\build\bin\Debug\llama-cli.exe"  # si lo usas
GGUF_PATH = r"C:\Users\TU_USUARIO\llama.cpp\build\bin\Debug\tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

BASE = Path(r"C:\Users\TU_USUARIO\Desktop\Llama_pruebas")  # raíz donde están los Marian

# Ajustes de rendimiento (puedes variarlos rápido desde aquí)
LLAMA_THREADS = "4"     # i7-7600U suele ir bien con 4
LLAMA_MAX_NEW = "64"    # baja a 32 si aún tarda
LLAMA_CTX = "256"       # suficiente para prompts cortos
LLAMA_UBATCH = "32"     # micro-batch para CPU
LLAMA_TIMEOUT = 45      # segundos

# Inglés -> Español
MAR_EN_ES_DIR = BASE / "opus-mt-en-es" / "opus-mt-en-es"
# Español -> Inglés (si lo descargas, quedará igual)
MAR_ES_EN_DIR = BASE / "opus-mt-es-en" / "opus-mt-es-en"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"device = {device}")

# --- Carga en-es (existe) ---
assert MAR_EN_ES_DIR.is_dir(), f"No existe la carpeta del modelo en-es: {MAR_EN_ES_DIR}"
tok_en_es = MarianTokenizer.from_pretrained(str(MAR_EN_ES_DIR))
mdl_en_es = MarianMTModel.from_pretrained(str(MAR_EN_ES_DIR)).to(device).eval()
log.info("Cargado traductor EN->ES")

# --- Carga es-en (opcional) ---
tok_es_en = mdl_es_en = None
if MAR_ES_EN_DIR.is_dir():
    tok_es_en = MarianTokenizer.from_pretrained(str(MAR_ES_EN_DIR))
    mdl_es_en = MarianMTModel.from_pretrained(str(MAR_ES_EN_DIR)).to(device).eval()
    log.info("Cargado traductor ES->EN")
else:
    log.warning("No se encontró traductor ES->EN (continuamos sin él).")


def translate(text: str, direction: str) -> str:
    """direction: 'es-en' o 'en-es'"""
    req = getattr(g, "req_id", "-")
    if not text.strip():
        log.info(f"[{req}] [TRADUCE {direction}] texto vacío")
        return ""

    if direction == "es-en":
        tok, mdl = tok_es_en, mdl_es_en
        if tok is None or mdl is None:
            log.info(f"[{req}] [TRADUCE {direction}] modelo no disponible; retorno original")
            return text
    else:
        tok, mdl = tok_en_es, mdl_en_es

    t0 = time.time()
    paras = [p for p in text.split("\n\n") if p.strip()]
    outs = []
    for p in paras:
        inputs = tok([p], return_tensors="pt", padding=True, truncation=True).to(device)
        gen = mdl.generate(**inputs, max_length=512)
        detok = tok.batch_decode(gen, skip_special_tokens=True)[0]
        outs.append(detok)
    dt = time.time() - t0
    log.info(f"[{req}] [TRADUCE {direction}] bloques={len(paras)} tiempo={dt:.2f}s")
    return "\n\n".join(outs)


def detect_es(text: str) -> bool:
    """Heurística simple para ES."""
    req = getattr(g, "req_id", "-")
    t = text.lower()
    if any(ch in t for ch in "áéíóúñ¿¡"):
        log.debug(f"[{req}] [DETECT] tilde/ñ detectadas -> ES")
        return True
    hits = sum(w in t for w in ["qué", "como", "dónde", "por qué", "cuál", "hola", "gracias", "capital", "país"])
    log.debug(f"[{req}] [DETECT] stopwords_es={hits}")
    return hits >= 2


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def run_llama(cmd, timeout_s=75):
    logging.info("[LLAMA CMD] %s", " ".join(shlex.quote(x) for x in cmd))
    t0 = time.time()
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # mezclamos stderr->stdout
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        return f"[llama timeout] superó los {timeout_s}s"

    dt = time.time() - t0
    logging.info("[LLAMA] latencia=%.2fs", dt)

    out = (res.stdout or "").strip()
    if res.returncode != 0:
        # devolvemos el texto de error tal cual para verlo en el browser
        return f"[error de llama] {out}"
    return out

LLAMA_THREADS_INT = 4          # i7-7600U: 4 hilos va bien
LLAMA_CTX_INT     = 1024        # como en tu prueba fuera de Flask
LLAMA_MAX_NEW_INT = 256

# Cargar el modelo UNA vez (tarda solo al levantar Flask)
log.info("Cargando TinyLlama GGUF en memoria (una sola vez)...")
LLM = Llama(
    model_path=GGUF_PATH,
    n_ctx=LLAMA_CTX_INT,
    n_threads=LLAMA_THREADS_INT,
    logits_all=False,
    verbose=False,     # menos ruido en consola
)

def call_llama(question_en: str) -> str:
    rid = getattr(g, "req_id", "-")
    t0 = time.time()
    try:
        # 1) completion "plain"
        res = LLM.create_completion(
            prompt=question_en,
            max_tokens=LLAMA_MAX_NEW_INT,
            temperature=0.9,
            top_p=0.95,
            top_k=40,
            stop=None,
            echo=False,
        )
        log.debug(f"[{rid}] [LLAMA raw completion] {str(res)[:300]}...")

        txt = ""
        if isinstance(res, dict):
            ch = (res.get("choices") or [])
            if ch:
                txt = (ch[0].get("text") or "").strip()
        elif hasattr(res, "choices"):
            txt = (res.choices[0].text or "").strip()
        else:
            txt = (str(res) or "").strip()

        # 2) si quedó vacío, probamos chat-completion (algunos modelos chat prefieren esto)
        if not txt:
            res2 = LLM.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",   "content": question_en},
                ],
                temperature=0.8,
                top_p=0.95,
                max_tokens=LLAMA_MAX_NEW_INT,
            )
            log.debug(f"[{rid}] [LLAMA raw chat] {str(res2)[:300]}...")
            if isinstance(res2, dict):
                ch2 = (res2.get("choices") or [])
                if ch2:
                    msg = ch2[0].get("message") or {}
                    txt = (msg.get("content") or "").strip()

        dt = time.time() - t0
        log.info(f"[{rid}] [LLAMA] latencia={dt:.2f}s  tokens={LLAMA_MAX_NEW_INT}")

        return txt or "[llama vacío]"
    except Exception:
        log.exception("Error generando con TinyLlama")
        return "[llama error]"
        

# === Flask app ===

app = Flask(__name__)
app.config.update(
    PROPAGATE_EXCEPTIONS=True,   # muestra tracebacks en consola
    JSONIFY_PRETTYPRINT_REGULAR=False
)
HTML = """
<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <title>TinyLlama + Traducción (local)</title>
    <style>
      body { font: 16px/1.4 system-ui, sans-serif; margin: 2rem; }
      textarea { width: 100%; height: 120px; }
      .out { white-space: pre-wrap; background:#f6f6f6; padding:1rem; border-radius:8px; }
      button { padding:.6rem 1rem; }
    </style>
  </head>
  <body>
    <h2>Pregunta (ES o EN)</h2>
    <textarea id="q" placeholder="Escribe tu pregunta..."></textarea>
    <br/><br/>
    <button onclick="ask()">Enviar</button>
    <h3>Respuesta (ES)</h3>
    <div id="a" class="out"></div>
    <script>
      async function ask(){
        const q = document.getElementById('q').value;
        document.getElementById('a').textContent = "Procesando...";
        const r = await fetch('/ask', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({q})});
        const j = await r.json();
        document.getElementById('a').textContent = j.answer_es || j.error || '(sin respuesta)';
      }
    </script>
  </body>
</html>
"""

# ===== Hooks para request_id y logs HTTP =====
@app.before_request
def _before():
    g.req_id = uuid.uuid4().hex[:8]
    try:
        payload = request.get_json(silent=True)
    except Exception:
        payload = None
    log.info(f"[{g.req_id}] --> {request.method} {request.path}  json={json.dumps(payload, ensure_ascii=False) if payload else None}")

@app.after_request
def _after(resp):
    log.info(f"[{g.req_id}] <-- {request.method} {request.path}  status={resp.status_code} len={resp.calculate_content_length()}")
    return resp


@app.route("/")
def index():
    # GET /
    return render_template_string(HTML)


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("q", "")
        t_all = time.time()
        data = request.get_json(force=True) or {}
        user_q = (data.get("q") or "").strip()
        log.info(f"[{g.req_id}] [ASK] q='{user_q[:120]}'")
        if not user_q:
            return jsonify(error="Pregunta vacía"), 400

        # 1) detectar idioma
        is_es = detect_es(user_q)
        log.info(f"[{g.req_id}] [ASK] detect_es={is_es}")

        # 2) traducir a inglés si vino en español
        t0 = time.time()
        if is_es and tok_es_en and mdl_es_en:
            q_en = translate(user_q, "es-en")
        else:
            q_en = user_q
        log.info(f"[{g.req_id}] [ASK] a) traducir es->en {time.time()-t0:.2f}s  prompt_en='{q_en[:120]}'")

        # 3) preguntar a TinyLlama (inglés)
        t1 = time.time()
        ans_en_raw = call_llama(q_en)
        log.info(f"[{g.req_id}] [ASK] b) llama {time.time()-t1:.2f}s  raw_len={len(ans_en_raw)}  raw_preview='{ans_en_raw[:120]}'")

        # 4) limpiar y traducir a español
        ans_en = (ans_en_raw or "").strip()
        if not ans_en or ans_en in ("[llama vacío]", "[llama error]"):
            return jsonify(error=f"Sin respuesta del modelo: {ans_en}"), 500

        t2 = time.time()
        ans_es = translate(ans_en, "en-es")
        log.info(f"[{g.req_id}] [ASK] c) traducir en->es {time.time()-t2:.2f}s")
        log.info(f"[{g.req_id}] [ASK] TOTAL {time.time()-t_all:.2f}s")

        return jsonify(question_in=user_q, prompt_en=q_en, answer_en=ans_en, answer_es=ans_es)
    
    except Exception as e:
        import traceback
        print("ERROR en /ask:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route("/health")
def health():
    try:
        out = call_llama("Say 'OK' if you are alive.")
        return {"ok": True, "llama": out[:200]}
    except Exception as e:
        return {"ok": False, "err": str(e)}, 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)