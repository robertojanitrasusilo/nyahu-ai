import gradio as gr
from config import APP_TITLE
from utils.models import panggil_nyahu_ai
import time

# Keluarkan dari fungsi main() dan jadikan variabel global bernama 'demo'

def predict(message, history):
    partial_message = ""
    for char in message:
        time.sleep(0.05)
        partial_message += char
        yield partial_message


demo = gr.ChatInterface(
    title=APP_TITLE,
    multimodal=True,
    save_history=True,
    fn=panggil_nyahu_ai,
    description='' + APP_TITLE + ' - Chatbot AI',
)
if __name__ == "__main__":
    demo.launch() 