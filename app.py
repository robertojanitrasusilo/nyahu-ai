import os

import gradio as gr
from config import APP_TITLE
from utils.models import panggil_nyahu_ai

demo = gr.ChatInterface(
    title=APP_TITLE,
    multimodal=True,
    save_history=True,
    fn=panggil_nyahu_ai,
    description='' + APP_TITLE + ' - AI ini dibuat karena muak dengan limit claude. Maka dibuatlah AI ini untuk membantu kalian semua. AI ini bisa menjawab pertanyaan kalian, membuatkan kode, dan lain-lain. Silahkan dicoba yaa.',
)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_port=port, server_name="0.0.0.0")
