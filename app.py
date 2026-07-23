import gradio as gr
from config import APP_TITLE
from utils.models import panggil_nyahu_ai

def main():
    # Setup the Gradio interface
    chat = gr.ChatInterface(
        fn=panggil_nyahu_ai,
        title=APP_TITLE,
        description="A highly capable assistant powered by Gemma 4 31B.",
        theme="soft",
        retry_btn="Retry",
        undo_btn="Undo",
        clear_btn="Clear",
    )

    # Launch local server
    chat.launch()

if __name__ == "__main__":
    main()
