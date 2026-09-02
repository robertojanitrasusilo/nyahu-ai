import uuid
import streamlit as st
from config import APP_TITLE
from utils.models import panggil_nyahu_ai

# 1. Konfigurasi Halaman & Theme
st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0rem; }
    .sub-header { color: #808495; font-size: 0.95rem; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# 2. Inisialisasi Session State Memori & Manajemen Sesi
# Pastikan dipaksa menjadi Dictionary {} agar tidak ada konflik memori lama
if "chat_sessions" not in st.session_state or not isinstance(st.session_state.chat_sessions, dict):
    st.session_state.chat_sessions = {}

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []


# SIDEBAR KHUSUS RIWAYAT  
with st.sidebar:
    
    
    # Tombol Obrolan Baru
    if st.button("Obrolan Baru", use_container_width=True, type="primary"):
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.session_state.document_chunks = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Sesi Percakapan:**")
    
    # Menampilkan daftar riwayat percakapan
    if st.session_state.chat_sessions:
        for s_id, s_data in reversed(list(st.session_state.chat_sessions.items())):
            is_active = (s_id == st.session_state.current_session_id)
            label = f"{'🟢' if is_active else '⚪'}{s_data['title']}"
            
            if st.button(label, key=f"sess_{s_id}", use_container_width=True):
                st.session_state.current_session_id = s_id
                st.session_state.messages = s_data["messages"].copy()
                st.rerun()
    else:
        st.caption("Belum ada riwayat.")

    st.markdown("<br>" * 2, unsafe_allow_html=True)
    if st.button("Hapus Semua Riwayat", use_container_width=True):
        st.session_state.chat_sessions = {}
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.session_state.document_chunks = []
        st.rerun()


#  MAIN INTERFACE 
st.markdown(f'<div class="main-header">{APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">AI Assistant RAG Hybrid (Web Search & PDF Local Turbovec).</div>', unsafe_allow_html=True)

# Render Pesan Chat Aktif
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# CHAT INPUT MULTIMODAL  
prompt_data = st.chat_input("Tanya sesuatu atau lampirkan dokumen...", accept_file="multiple")

if prompt_data:
    user_input = prompt_data.text if hasattr(prompt_data, "text") else str(prompt_data)
    uploaded_files = prompt_data.files if hasattr(prompt_data, "files") else []

    if user_input or uploaded_files: 
        curr_id = st.session_state.current_session_id
        
        # Buat sesi baru JIKA id belum ada ATAU id hilang dari memori dictionary
        if curr_id is None or curr_id not in st.session_state.chat_sessions:
            curr_id = str(uuid.uuid4())
            st.session_state.current_session_id = curr_id
            
            # Bikin judul dari 26 huruf pertama user (biar rapi di sidebar)
            judul_sesi = user_input[:26] + "..." if user_input else "Lampiran Berkas"
            
            # Daftarkan loker baru di memori
            st.session_state.chat_sessions[curr_id] = {
                "title": judul_sesi,
                "messages": []
            }
        

        with st.chat_message("user"):
            if uploaded_files:
                for f in uploaded_files:
                    st.caption(f"📎 *Lampiran: {f.name}*")
            if user_input:
                st.markdown(user_input)

        payload_file = None
        if uploaded_files:
            f_obj = uploaded_files[0]
            payload_file = {
                "name": f_obj.name,
                "type": f_obj.type,
                "bytes": f_obj.read()
            }

        msg_text = user_input or "[Mengirimkan Berkas]"
        st.session_state.messages.append({"role": "user", "content": msg_text})

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            for response_chunk in panggil_nyahu_ai(user_input, payload_file):
                full_response = response_chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Update history dengan aman pakai curr_id yang sudah divalidasi pasti ada
        st.session_state.chat_sessions[curr_id]["messages"] = st.session_state.messages.copy()
        
        # Rerun layar biar history barunya langsung muncul otomatis di sidebar
        st.rerun()