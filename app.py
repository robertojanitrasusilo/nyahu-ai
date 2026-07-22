import gradio as gr
from huggingface_hub import InferenceClient

# 1. Masukkan token Hugging Face lu di sini
hf_token = "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" 

# 2. Inisialisasi client API (menggunakan parameter api_key sesuai snippet terbaru)
client = InferenceClient(
    api_key=hf_token,
)

def panggil_gemma_hf(pesan, riwayat):
    messages = []
    
    # Looping riwayat dengan cara yang aman untuk Gradio versi terbaru
    for item in riwayat:
        # Pengecekan jika formatnya adalah dictionary (Gradio Baru)
        if isinstance(item, dict):
            # Ambil role dan content-nya, abaikan atribut lain seperti metadata
            messages.append({
                "role": item.get("role", "user"), 
                "content": item.get("content", "")
            })
        # Pengecekan *fallback* jika formatnya masih list/tuple 2 item (Gradio Lama)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            messages.append({"role": "user", "content": item[0]})
            messages.append({"role": "assistant", "content": item[1]})
    
    # Tambahkan pesan baru dari input pengguna
    messages.append({"role": "user", "content": pesan})
    
    try:
        completion = client.chat.completions.create(
            model="google/gemma-2b-it:featherless-ai", 
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            stream=True  # Menambahkan streaming jika diperlukan
        )
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"Wah, ada error saat mengeksekusi model nih: {str(e)}"

    
# 7. Bangun antarmuka Gradio
app = gr.ChatInterface(
    fn=panggil_gemma_hf,
    title="Gemma API Chat (Featherless AI)",
    description="Menjalankan Gemma 2B via endpoint Featherless di Hugging Face.",
)

if __name__ == "__main__":
    # Langsung jalankan aplikasinya
    app.launch()