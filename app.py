import os

import gradio as gr
from config import APP_TITLE
from utils.models import panggil_nyahu_ai, generate_chat_title




# Header HTML
header_html = f"""
<div style="text-align: center; margin-bottom: 15px;">
    <h1 style="font-size: 2.2rem; font-weight: 800; background: linear-gradient(90deg, #A855F7, #06B6D4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 4px;">
         {APP_TITLE}
    </h1>
    <p style="color: #94A3B8; font-size: 0.85rem; font-family: monospace;">
        SYSTEM ONLINE | GEMMA 4 REASONING & VISION ENGINE
    </p>
</div>
"""

# Menggunakan Theme Dark Gradio
theme = gr.themes.Base(
    primary_hue="purple",
    secondary_hue="cyan",
    neutral_hue="slate",
).set(
    body_background_fill="#090D16",
    block_background_fill="#0F172A",
    block_border_width="1px",
    block_border_color="#1E293B",
)

with gr.Blocks(theme=theme, css_paths=["style.css"] if os.path.exists("style.css") else []) as demo:
    gr.HTML(header_html)

    # State untuk menyimpan riwayat percakapan & judul-judul sesi
    # sessions_state: dict -> {session_id: {"title": str, "messages": list}}
    sessions_state = gr.State(value={})
    active_session_id = gr.State(value="session_1")

    with gr.Row():
        # Sidebar Kiri (Riwayat Chat & Tombol New Chat)
        with gr.Column(scale=1, min_width=260):
            new_chat_btn = gr.Button("➕ New Chat", variant="primary")
            gr.Markdown("### 📜 Riwayat Sesi")
            history_dropdown = gr.Radio(
                choices=[],
                label="Pilih Obrolan",
                interactive=True,
                show_label=False
            )

        # Area Obrolan Utama
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height=520,
                avatar_images=(None, None)
            )
            
            with gr.Row():
                user_input = gr.MultimodalTextbox(
                    placeholder="Ketik pesan atau paste/upload gambar & PDF...",
                    show_label=False,
                    scale=8,
                    file_types=["image", ".pdf"]
                )
                submit_btn = gr.Button("Kirim ", variant="primary", scale=1)

            with gr.Row():
                clear_btn = gr.Button("🗑 Clear Current Chat", variant="secondary")

    # ================= EVENT HANDLERS / LOGIC ================= #

    def bot_respond(user_data, history, sessions, current_id):
        """
        Mengeksekusi jawaban AI secara streaming,
        sekaligus generate Judul AI Otomatis di pesan pertama.
        """
        if not user_data or (isinstance(user_data, dict) and not user_data.get("text") and not user_data.get("files")):
            yield history, user_data, sessions, gr.update()
            return

        # Ambil teks mentah untuk penentuan judul
        raw_text = ""
        if isinstance(user_data, dict):
            raw_text = user_data.get("text", "")
            if not raw_text and user_data.get("files"):
                raw_text = os.path.basename(user_data["files"][0])
        else:
            raw_text = str(user_data)

        # Jika sesi baru belum punya pesan, buatkan Judul Otomatis pakai AI
        is_first_message = len(history) == 0
        
        # Tambahkan input user ke tampilan chat sementara
        history_for_api = list(history)
        
        # Panggil API AI secara streaming
        generator = panggil_nyahu_ai(user_data, history_for_api)
        
        # Inisialisasi bubble respons bot
        history.append({"role": "user", "content": raw_text})
        history.append({"role": "assistant", "content": ""})

        for partial_response in generator:
            history[-1]["content"] = partial_response
            yield history, gr.update(value=None, interactive=True), sessions, gr.update()

        # Update simpanan di state
        if current_id not in sessions:
            sessions[current_id] = {"title": "Obrolan Baru", "messages": []}

        # Generate judul dinamis jika ini pesan pertama
        if is_first_message:
            ai_title = generate_chat_title(raw_text)
            sessions[current_id]["title"] = ai_title

        sessions[current_id]["messages"] = history

        # Update pilihan dropdown di sidebar
        dropdown_choices = [(data["title"], sid) for sid, data in sessions.items()]
        
        yield history, gr.update(value=None, interactive=True), sessions, gr.update(choices=dropdown_choices, value=current_id)

    def start_new_chat(sessions):
        """Membuat sesi chat baru."""
        new_id = f"session_{len(sessions) + 1}"
        sessions[new_id] = {"title": "Obrolan Baru", "messages": []}
        
        dropdown_choices = [(data["title"], sid) for sid, data in sessions.items()]
        return [], new_id, sessions, gr.update(choices=dropdown_choices, value=new_id)

    def switch_chat(selected_session_id, sessions):
        """Berpindah ke sesi chat lain saat memilih di sidebar."""
        if selected_session_id in sessions:
            return sessions[selected_session_id]["messages"], selected_session_id
        return [], selected_session_id

    def clear_current(sessions, current_id):
        """Clear pesan di sesi aktif."""
        if current_id in sessions:
            sessions[current_id]["messages"] = []
        return [], sessions

    # Connect Events
    submit_event = submit_btn.click(
        fn=bot_respond,
        inputs=[user_input, chatbot, sessions_state, active_session_id],
        outputs=[chatbot, user_input, sessions_state, history_dropdown]
    )

    user_input.submit(
        fn=bot_respond,
        inputs=[user_input, chatbot, sessions_state, active_session_id],
        outputs=[chatbot, user_input, sessions_state, history_dropdown]
    )

    new_chat_btn.click(
        fn=start_new_chat,
        inputs=[sessions_state],
        outputs=[chatbot, active_session_id, sessions_state, history_dropdown]
    )

    history_dropdown.change(
        fn=switch_chat,
        inputs=[history_dropdown, sessions_state],
        outputs=[chatbot, active_session_id]
    )

    clear_btn.click(
        fn=clear_current,
        inputs=[sessions_state, active_session_id],
        outputs=[chatbot, sessions_state]
    )

if __name__ == "__main__":
    demo.launch()