import torch
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# -------------------------
# Hardcoded config
# -------------------------
BASE_MODEL_ID = "EleutherAI/gpt-neo-2.7B"      # base Falcon
LORA_HF_ID = "sedaghat989Dev/floria-lora-2.0"         # Floria LoRA
MODEL_CACHE_DIR = "./models"                     # local cache
MAX_NEW_TOKENS = 150

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__, template_folder="templates")

# -------------------------
# Device detection
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[init] device = {device}")

# -------------------------
# BitsAndBytes 4-bit config if GPU
# -------------------------
bnb_config = None
if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"
    )

# -------------------------
# Load tokenizer
# -------------------------
print("[init] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, cache_dir=MODEL_CACHE_DIR)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# Load Falcon
# -------------------------
print("[init] Loading base Falcon model...")
if bnb_config:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=MODEL_CACHE_DIR
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto" if device=="cuda" else None,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
        cache_dir=MODEL_CACHE_DIR
    )
print("[init] Base model loaded.")

# -------------------------
# Apply hardcoded LoRA
# -------------------------
print(f"[init] Applying Floria LoRA from {LORA_HF_ID}...")
model = PeftModel.from_pretrained(model, LORA_HF_ID, device_map="auto")
model.eval()
print("[init] LoRA applied. Model ready.")

# -------------------------
# Helper: generate reply
# -------------------------
def generate_reply(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):].strip()
    return decoded

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = data.get("message", "")
    if not isinstance(user_message, str):
        return jsonify({"error": "invalid message"}), 400

    prompt = f"User: {user_message}\nFloria:"
    try:
        reply = generate_reply(prompt)
    except Exception as e:
        print("[error] generation failed:", str(e))
        return jsonify({"error": "generation failed"}), 500

    return jsonify({"user": user_message, "floria": reply})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": device})

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
