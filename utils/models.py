import os
import base64
import mimetypes
import numpy as np
from datetime import datetime
from openai import OpenAI
from tavily import TavilyClient

# Import RAG Stack Lokal
from turbovec import TurboQuantIndex
from sentence_transformers import SentenceTransformer

from config import OPENROUTER_API_KEY, TAVILY_API_KEY, MODEL_NAME, APP_TITLE

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

tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

# init turbovec index dan sentence transformeryang dipake
print("Memuat model embedding lokal (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Model all-MiniLM-L6-v2 memiliki dimensi 384. Kita pakai 4-bit quantization untuk hemat RAM.
vector_index = TurboQuantIndex(dim=384, bit_width=4)
document_chunks = [] # Tempat menyimpan teks asli sesuai urutan indeks Turbovec


def get_system_prompt() -> str:
    """Membaca system prompt dan menyuntikkan waktu saat ini agar AI sadar waktu."""
    file_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'system.txt')
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            base_prompt = f.read().strip()
    except Exception:
        base_prompt = "You are Nyahu AI, a highly capable assistant."

    sekarang = datetime.now().strftime("%d %B %Y")
    tahun_ini = datetime.now().year
    
    time_awareness = f"""

[INSTRUKSI MUTLAK SISTEM]: 
Hari ini adalah tanggal {sekarang}. Kamu sudah berada di tahun {tahun_ini}. 
Jangan pernah berkata bahwa kamu "tidak bisa meramal masa depan" jika ditanya tentang tahun {tahun_ini} dan sebelumnya. Gunakan fakta dari RAG Search (Web maupun Dokumen Lokal) yang dilampirkan pengguna secara penuh!
"""
    return base_prompt + time_awareness


def get_safe_path(obj) -> str:
    """Mengekstrak file path secara aman dari objek Gradio."""
    if isinstance(obj, dict):
        return obj.get("path", obj.get("name", ""))
    elif hasattr(obj, "path"):
        return obj.path
    return str(obj)


def encode_image_to_base64(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/png"
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"


def cari_internet_tavily(query: str, max_results: int = 3) -> str:
    """Melakukan pencarian RAG tingkat lanjut menggunakan Tavily API."""
    if not query or len(query.strip()) < 3 or not tavily_client:
        return ""
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True
        )
        formatted_results = []
        if response.get("answer"):
            formatted_results.append(f"[Tavily AI Insight]: {response['answer']}\n")
        for i, r in enumerate(response.get("results", []), 1):
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            formatted_results.append(f"[{i}] {title}\nSumber: {url}\nKonten: {content}")
        return "\n\n".join(formatted_results)
    except Exception as e:
        print(f"[Warning] Gagal melakukan pencarian Tavily: {e}")
        return ""


def panggil_nyahu_ai(pesan, riwayat: list):
    system_prompt = get_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]

    # 1. Parsing Riwayat (Aman untuk Multimodal Gradio 6)
    for item in riwayat:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if isinstance(content, (list, tuple)):
                path_str = get_safe_path(content[0]) if content else ""
                content = f"[User melampirkan file: {os.path.basename(path_str)}]" if path_str else "[File Terlampir]"
            elif type(content).__name__ == "FileData" or hasattr(content, "path"):
                path_str = get_safe_path(content)
                content = f"[User melampirkan file: {os.path.basename(path_str)}]"
            elif isinstance(content, dict) and ("path" in content or "name" in content):
                path_str = get_safe_path(content)
                content = f"[User melampirkan file: {os.path.basename(path_str)}]"

            if role and content:
                messages.append({"role": role, "content": str(content)})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, assistant_msg = item
            if user_msg:
                messages.append({"role": "user", "content": str(user_msg)})
            if assistant_msg:
                messages.append({"role": "assistant", "content": str(assistant_msg)})

    user_text_parts = []
    user_image_parts = []
    prompt_text = ""

    # 2. Parsing Input Pesan Baru
    if isinstance(pesan, dict):
        prompt_text = pesan.get("text") or ""
        if prompt_text:
            user_text_parts.append(prompt_text)

        files = pesan.get("files") or []
        for file_obj in files:
            file_path = get_safe_path(file_obj)
            ext = os.path.splitext(file_path)[1].lower()
            nama_file_saja = os.path.basename(file_path)

            if ext == ".pdf":
                if HAS_PYPDF:
                    reader = PdfReader(file_path)
                    full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    
                    # RAG INGESTION: Potong teks jadi chunks (per 1000 huruf)
                    chunk_size = 1000
                    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size) if full_text[i:i+chunk_size].strip()]
                    
                    if chunks:
                        # Encode pakai Sentence Transformers lalu masukkan ke Turbovec
                        vectors = embedder.encode(chunks).astype(np.float32)
                        vector_index.add(vectors)
                        document_chunks.extend(chunks)
                        user_text_parts.append(f"\n\n[Sistem: Dokumen '{nama_file_saja}' diproses. {len(chunks)} bagian berhasil disuntikkan ke memori Turbovec.]")
                else:
                    user_text_parts.append(f"\n\n[Error: Library 'pypdf' belum diinstall.]")
                    
            elif ext in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"]:
                base64_img = encode_image_to_base64(file_path)
                user_image_parts.append({"type": "image_url", "image_url": {"url": base64_img}})
    else:
        prompt_text = str(pesan)
        user_text_parts.append(prompt_text)

    # 1. RAG LOKAL (TURBOVEC DOKUMEN)
    if prompt_text and len(document_chunks) > 0:
        query_vec = embedder.encode([prompt_text]).astype(np.float32)
        # Cari 3 potongan teks PDF paling relevan
        scores, indices = vector_index.search(query_vec, k=3) 
        
        retrieved_texts = [document_chunks[i] for i in indices[0] if i < len(document_chunks)]
        if retrieved_texts:
            konteks_lokal = "\n\n".join([f"[Kutipan PDF {idx+1}] {txt}" for idx, txt in enumerate(retrieved_texts)])
            user_text_parts.append(
                f"\n\n--- DATA REFERENSI DOKUMEN PDF LOKAL ---\n{konteks_lokal}\n---------------------------------------\n"
            )

    # 2. RAG WEB (TAVILY API) 
    if prompt_text and not user_image_parts:
        hasil_pencarian = cari_internet_tavily(prompt_text, max_results=3)
        if hasil_pencarian:
            user_text_parts.append(
                f"\n\n--- DATA REFERENSI WEB (TAVILY) ---\n{hasil_pencarian}\n---------------------------------------\n"
            )

    # ================= INSTRUKSI FINAL ================= #
    if len(document_chunks) > 0 or (prompt_text and not user_image_parts and tavily_client):
         user_text_parts.append("[INSTRUKSI AI]: Jawab berdasarkan referensi PDF Lokal atau Web di atas. Utamakan PDF jika pertanyaan berkaitan dengan dokumen!")

    combined_text_prompt = "\n".join(user_text_parts).strip()
    if not combined_text_prompt and user_image_parts:
        combined_text_prompt = "Tolong analisis dan jelaskan gambar yang saya lampirkan ini."

    user_content = []
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