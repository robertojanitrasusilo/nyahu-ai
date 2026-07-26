import os
import asyncio
from nicegui import ui, app
from config import APP_TITLE
from utils.models import panggil_nyahu_ai, generate_chat_title

# State Global Sesi
# Structure: {session_id: {"title": str, "messages": list}}
sessions = {"session_1": {"title": "Obrolan Baru", "messages": []}}
current_session_id = "session_1"
uploaded_files = []

# Terapkan Custom CSS (Design Solid Flat ala Google Stitch / ChatGPT)
if os.path.exists("style.css"):
    app.add_static_files('/static', '.')
    ui.add_head_html('<link rel="stylesheet" href="/static/style.css">', shared=True)

@ui.page('/')
def main():
    global current_session_id, uploaded_files
    
    # Body background styling (Solid Dark Slate)
    ui.query('body').style('background-color: #0F172A; color: #F8FAFC; font-family: "Inter", sans-serif;')

    # Layout Utama: Grid 2 Kolom (Sidebar Kiri & Main Chat Kanan)
    with ui.row().classes('w-full h-screen no-wrap gap-0'):
        
        # ================= SIDEBAR KIRI ================= #
        with ui.column().classes('w-72 h-full bg-[#1E293B] p-4 flex flex-col justify-between border-r border-[#334155]'):
            with ui.column().classes('w-full gap-3'):
                # Header Logo / Title
                with ui.row().classes('items-center gap-2 mb-2'):
                    ui.icon('smart_toy', color='blue-600').classes('text-2xl')
                    ui.label(APP_TITLE).classes('text-xl font-bold text-white tracking-wide')
                
                # Tombol New Chat (Solid Blue)
                new_chat_btn = ui.button('➕ New Chat', on_click=lambda: create_new_chat())\
                    .classes('w-full bg-[#2563EB] text-white font-semibold py-2 rounded-lg no-shadow hover:bg-[#1D4ED8] transition-all')
                
                ui.label('RECENTS').classes('text-xs font-bold text-[#94A3B8] mt-4 tracking-wider')
                
                # Container Daftar Riwayat Chat
                history_container = ui.column().classes('w-full gap-1 overflow-y-auto max-h-[60vh]')

        # ================= AREA CHAT UTAMA ================= #
        with ui.column().classes('flex-1 h-full bg-[#0F172A] p-6 flex flex-col justify-between overflow-hidden'):
            
            # Header Chat Active Title
            with ui.row().classes('w-full pb-3 border-b border-[#1E293B] items-center justify-between'):
                active_title_label = ui.label('Obrolan Baru').classes('text-lg font-semibold text-white')
                ui.label('Gemma 4 Reasoning Engine').classes('text-xs text-[#64748B] font-mono')

            # Container Messages Scrollable
            chat_scroll = ui.scroll_area().classes('w-full flex-1 my-4 pr-2')
            with chat_scroll:
                messages_container = ui.column().classes('w-full max-w-4xl mx-auto gap-4')

            # Preview Attachment File
            file_preview_container = ui.row().classes('w-full max-w-4xl mx-auto gap-2 mb-1')

            # Area Input Bar
            with ui.column().classes('w-full max-w-4xl mx-auto gap-2'):
                with ui.row().classes('w-full items-center bg-[#1E293B] rounded-xl p-2 border border-[#334155] shadow-sm'):
                    
                    # Upload Button
                    ui.upload(
                        on_upload=lambda e: handle_upload(e),
                        multiple=True,
                        auto_upload=True
                    ).props('flat round dense icon=attach_file').classes('text-[#94A3B8]')

                    # Input Field Textarea
                    input_field = ui.textarea(placeholder=f'Message {APP_TITLE}...')\
                        .props('borderless dense autogrow rows=1')\
                        .classes('flex-1 text-white px-3 bg-transparent text-sm focus:outline-none')
                    
                    # Send Button
                    send_btn = ui.button(icon='send', on_click=lambda: send_message())\
                        .props('flat round dense').classes('text-[#2563EB]')

                ui.label(f'{APP_TITLE} can make mistakes. Verify important info.').classes('text-[11px] text-[#64748B] text-center w-full')

    # ================= LOGIKA EVENT & FUNCTIONS ================= #

    def update_sidebar():
        """Memperbarui daftar item riwayat obrolan di sidebar."""
        history_container.clear()
        with history_container:
            for sid, data in sessions.items():
                is_active = (sid == current_session_id)
                btn_cls = 'w-full text-left justify-start py-2 px-3 rounded-md text-xs truncate '
                btn_cls += 'bg-[#334155] text-white font-medium' if is_active else 'text-[#94A3B8] hover:bg-[#1E293B] hover:text-white'
                
                ui.button(data["title"], on_click=lambda s=sid: switch_session(s)).classes(btn_cls).props('flat no-caps')

    def switch_session(sid):
        """Berpindah sesi chat."""
        global current_session_id
        current_session_id = sid
        active_title_label.set_text(sessions[sid]["title"])
        update_sidebar()
        render_messages()

    def create_new_chat():
        """Membuat sesi baru."""
        global current_session_id
        new_id = f"session_{len(sessions) + 1}"
        sessions[new_id] = {"title": "Obrolan Baru", "messages": []}
        switch_session(new_id)

    def handle_upload(e):
        """Menangani file yang diunggah."""
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, e.name)
        
        with open(file_path, 'wb') as f:
            f.write(e.content.read())
            
        uploaded_files.append(file_path)
        render_file_previews()

    def render_file_previews():
        """Menampilkan chip indikator file yang diunggah."""
        file_preview_container.clear()
        with file_preview_container:
            for fpath in uploaded_files:
                fname = os.path.basename(fpath)
                with ui.row().classes('bg-[#334155] text-xs text-white px-2 py-1 rounded-md items-center gap-1'):
                    ui.icon('insert_drive_file').classes('text-xs')
                    ui.label(fname).classes('max-w-[150px] truncate')

    def render_messages():
        """Render ulang seluruh gelembung obrolan untuk sesi aktif."""
        messages_container.clear()
        current_msgs = sessions[current_session_id]["messages"]
        with messages_container:
            for msg in current_msgs:
                role = msg.get("role")
                content = msg.get("content", "")
                
                if role == "user":
                    ui.chat_message(content, name='You', sent=True).classes('w-full').props('bg-color=blue-7 text-color=white')
                elif role == "assistant":
                    ui.chat_message(content, name=APP_TITLE, sent=False).classes('w-full').props('bg-color=slate-8 text-color=white')

    async def send_message():
        """Mengeksekusi pengiriman pesan & streaming balasan AI."""
        global current_session_id, uploaded_files
        
        text = input_field.value.strip()
        if not text and not uploaded_files:
            return

        # Rakit payload
        payload = {"text": text, "files": list(uploaded_files)}
        
        # Simpan teks asli
        user_display_text = text if text else (os.path.basename(uploaded_files[0]) if uploaded_files else "")
        
        # Reset Input
        input_field.value = ''
        uploaded_files.clear()
        render_file_previews()

        session = sessions[current_session_id]
        is_first = len(session["messages"]) == 0

        # Tambahkan pesan user ke state
        session["messages"].append({"role": "user", "content": user_display_text})
        
        # Tambahkan placeholder balasan assistant
        session["messages"].append({"role": "assistant", "content": ""})
        render_messages()

        # Generate Judul Otomatis jika pesan pertama
        if is_first:
            ai_title = generate_chat_title(user_display_text)
            session["title"] = ai_title
            active_title_label.set_text(ai_title)
            update_sidebar()

        # Panggil generator API OpenRouter secara async streaming
        history_for_api = session["messages"][:-2]
        gen = panggil_nyahu_ai(payload, history_for_api)

        # Loop generator di thread terpisah agar UI NiceGUI tetap responsif
        while True:
            chunk = await asyncio.to_thread(next, gen, None)
            if chunk is None:
                break
            
            session["messages"][-1]["content"] = chunk
            render_messages()
            chat_scroll.scroll_to(percent=1.0)  # <-- Hapus 'await' di sini
            await asyncio.sleep(0.01)

    # Inisialisasi awal
    update_sidebar()
    render_messages()

# Jalankan NiceGUI
ui.run(host='0.0.0.0', port=7860, title=APP_TITLE, reload=False)