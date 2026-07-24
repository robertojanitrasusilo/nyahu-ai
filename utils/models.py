import os
import base64
import mimetypes
from openai import OpenAI
from config import OPENROUTER_API_KEY, MODEL_NAME, APP_TITLE

# Inisialisasi client OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "http://localhost:7860",
        "X-Title": APP_TITLE
    }
)

def get_system_prompt() -> str:
    """Membaca system prompt dari file."""
    file_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'system.txt')
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "You are Nyahu AI, a highly capable assistant."

def encode_image_to_base64(image_path: str) -> str:
    """Mengubah file gambar lokal menjadi format Data URL Base64."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/png"
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"

def panggil_nyahu_ai(pesan, riwayat: list):
    """
    Fungsi Generator Gradio ChatInterface dengan fitur Streaming & Multimodal.
    """
    system_prompt = get_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]

    # 1. Parsing riwayat obrolan terdahulu
    for item in riwayat:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role and content:
                messages.append({"role": role, "content": content})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, assistant_msg = item
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})

    # 2. Parsing input terbaru dari user (Multimodal Handling)
    user_content = []

    if isinstance(pesan, dict):
        text_content = pesan.get("text", "")
        if text_content:
            user_content.append({"type": "text", "text": text_content})

        files = pesan.get("files", [])
        for file_path in files:
            base64_image = encode_image_to_base64(file_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": base64_image}
            })
    else:
        user_content = pesan

    messages.append({"role": "user", "content": user_content})

    # 3. Eksekusi pemanggilan streaming ke api
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True,  
            extra_body={
                "reasoning": {
                    "enabled": True
                }
            }
        )

        partial_text = ""
        # Loop tiap potongan teks (chunk) yang masuk dari OpenRouter
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", "") or ""
                
                if content:
                    partial_text += content
                    yield partial_text  # kirim teks parsial ke Gradio ChatInterface secara streaming

    except Exception as e:
        yield f"Terjadi kesalahan saat menghubungkan ke Nyahu AI: {str(e)}" 