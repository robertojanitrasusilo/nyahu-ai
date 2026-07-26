import os
import base64
import mimetypes
from openai import OpenAI
from config import OPENROUTER_API_KEY, MODEL_NAME, APP_TITLE

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "http://localhost:7860",
        "X-Title": APP_TITLE
    }
)

def get_system_prompt() -> str:
    file_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'system.txt')
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "You are Nyahu AI, a highly capable assistant."

def encode_image_to_base64(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/png"
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"

def extract_text_from_pdf(pdf_path: str) -> str:
    if not HAS_PYPDF:
        return "[Error: Library 'pypdf' belum diinstall.]"
    try:
        reader = PdfReader(pdf_path)
        extracted_text = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                extracted_text.append(f"--- Halaman {index} ---\n{text}")
        return "\n\n".join(extracted_text) if extracted_text else "[PDF Kosong/Scan Image]"
    except Exception as e:
        return f"[Gagal membaca PDF: {str(e)}]"

def panggil_nyahu_ai(pesan, riwayat: list):
    system_prompt = get_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]

    for item in riwayat:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role and content:
                messages.append({"role": role, "content": content})

    user_text_parts = []
    user_image_parts = []

    if isinstance(pesan, dict):
        prompt_text = pesan.get("text", "")
        if prompt_text:
            user_text_parts.append(prompt_text)

        files = pesan.get("files", [])
        for file_path in files:
            ext = os.path.splitext(file_path)[1].lower()
            nama_file_saja = os.path.basename(file_path)

            if ext == ".pdf":
                pdf_content = extract_text_from_pdf(file_path)
                user_text_parts.append(f"\n\n--- ISIDOKUMEN PDF: {nama_file_saja} ---\n{pdf_content}\n--- AKHIR DOKUMEN PDF ---")
            elif ext in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"]:
                base64_img = encode_image_to_base64(file_path)
                user_image_parts.append({"type": "image_url", "image_url": {"url": base64_img}})
    else:
        user_text_parts.append(str(pesan))

    user_content = []
    combined_text_prompt = "\n".join(user_text_parts).strip()
    if combined_text_prompt:
        user_content.append({"type": "text", "text": combined_text_prompt})
    user_content.extend(user_image_parts)

    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            extra_body={"reasoning": {"enabled": True}}
        )

        partial_text = ""
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", "") or ""
                if content:
                    partial_text += content
                    yield partial_text
    except Exception as e:
        yield f"Terjadi kesalahan saat menghubungkan ke Nyahu AI: {str(e)}"

def generate_chat_title(first_message: str) -> str:
    if not first_message or not isinstance(first_message, str):
        return "Obrolan Baru"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Rangkum pesan pengguna menjadi judul singkat (3-5 kata). Jangan gunakan tanda petik atau penjelas."
                },
                {"role": "user", "content": first_message}
            ],
            max_tokens=15,
            temperature=0.5
        )
        title = response.choices[0].message.content.strip().replace('"', '').replace("'", "")
        return title if title else (first_message[:20] + "...")
    except Exception:
        return first_message[:20] + "..." if len(first_message) > 20 else first_message