import os
import base64
import mimetypes
import numpy as np
import streamlit as st
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
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": APP_TITLE
    }
)

tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

# OPTIMASI MEMORI STREAMLIT 
@st.cache_resource
def inisialisasi_rag_lokal():
    # Memuat model embedding dan index Turbovec sekali saja ke dalam memori cache.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Model all-MiniLM-L6-v2 memiliki dimensi 384. Pakai 4-bit quantization untuk hemat RAM.
    idx = TurboQuantIndex(dim=384, bit_width=4)
    return model, idx

embedder, vector_index = inisialisasi_rag_lokal()

# Gunakan session state Streamlit untuk menampung teks dokumen agar persisten
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

def get_system_prompt() -> str:
    # Membaca system prompt dan menyuntikkan waktu saat ini agar AI sadar waktu.
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
Jangan pernah berkata bahwa kamu "tidak bisa meramal masa depan" jika ditanya tentang tahun {tahun_ini} dan sebelumnya. Gunakan fakta dari RAG Search (Web maupun Dokumen Lokal) secara penuh!
"""
    return base_prompt + time_awareness


def encode_bytes_to_base64(file_bytes, mime_type: str) -> str:
    # Mengubah file bytes dari berkas unggahan langsung menjadi format Data URL Base64.
    encoded_string = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"


def proses_dokumen_pdf(file_obj, nama_file_saja: str):
    # Mengekstrak teks PDF dan mengindeks potongan ke memori Turbovec.
    if not HAS_PYPDF:
        return f"[Error: Library 'pypdf' belum diinstall.]"
    try:
        reader = PdfReader(file_obj)
        full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        chunk_size = 1000
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size) if full_text[i:i+chunk_size].strip()]
        
        if chunks:
            vectors = embedder.encode(chunks).astype(np.float32)
            vector_index.add(vectors)
            st.session_state.document_chunks.extend(chunks)
            return f"\n\n[Sistem: Dokumen '{nama_file_saja}' diproses. {len(chunks)} bagian berhasil disuntikkan ke memori Turbovec.]"
        return "\n\n[Sistem: Dokumen PDF kosong atau berupa pindaian gambar.]"
    except Exception as e:
        return f"[Gagal membaca PDF: {str(e)}]"


def cari_internet_tavily(query: str, max_results: int = 3) -> str:
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


def panggil_nyahu_ai(prompt_text: str, file_data: dict = None):
    """Fungsi handler RAG Hybrid untuk memproses prompt teks dan lampiran file ke OpenRouter."""
    system_prompt = get_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]

    # 1. Parsing Riwayat Obrolan dari State Streamlit
    for chat in st.session_state.messages:
        messages.append({"role": chat["role"], "content": chat["content"]})

    user_text_parts = [prompt_text]
    user_image_parts = []

    # 2. Pemrosesan Lampiran File
    if file_data:
        nama_file = file_data.get("name", "")
        mime_type = file_data.get("type", "")
        file_bytes = file_data.get("bytes")

        if nama_file.lower().endswith(".pdf"):
            # Karena berbentuk bytes dari Streamlit, bungkus atau baca langsung
            from io import BytesIO
            pdf_stream = BytesIO(file_bytes)
            status_msg = proses_dokumen_pdf(pdf_stream, nama_file)
            user_text_parts.append(status_msg)
            
        elif mime_type.startswith("image/"):
            base64_img = encode_bytes_to_base64(file_bytes, mime_type)
            user_image_parts.append({"type": "image_url", "image_url": {"url": base64_img}})

    # 3. Eksekusi RAG Lokal (Turbovec)
    if prompt_text and len(st.session_state.document_chunks) > 0:
        query_vec = embedder.encode([prompt_text]).astype(np.float32)
        scores, indices = vector_index.search(query_vec, k=3) 
        
        retrieved_texts = [st.session_state.document_chunks[i] for i in indices[0] if i < len(st.session_state.document_chunks)]
        if retrieved_texts:
            konteks_lokal = "\n\n".join([f"[Kutipan PDF {idx+1}] {txt}" for idx, txt in enumerate(retrieved_texts)])
            user_text_parts.append(
                f"\n\n--- DATA REFERENSI DOKUMEN PDF LOKAL \n{konteks_lokal}\n\n"
            )

    # 4. Eksekusi RAG Web (Tavily)
    if prompt_text and not user_image_parts:
        hasil_pencarian = cari_internet_tavily(prompt_text, max_results=3)
        if hasil_pencarian:
            user_text_parts.append(
                f"\n\n--- DATA REFERENSI WEB (TAVILY) \n{hasil_pencarian}\n\n"
            )

    # 5. Gabungkan Instruksi Akhir
    if len(st.session_state.document_chunks) > 0 or (prompt_text and not user_image_parts and tavily_client):
         user_text_parts.append("[INSTRUKSI AI]: Jawab berdasarkan referensi PDF Lokal atau Web di atas. Utamakan PDF jika pertanyaan berkaitan dengan dokumen!")

    combined_text_prompt = "\n".join(user_text_parts).strip()
    
    user_content = []
    if combined_text_prompt:
        user_content.append({"type": "text", "text": combined_text_prompt})
    user_content.extend(user_image_parts)
    
    # Tambahkan pesan user saat ini ke list payload API
    messages.append({"role": "user", "content": user_content})

    # 6. Streaming Jawaban dari OpenRouter
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