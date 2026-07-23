import os
from openai import OpenAI
from config import OPENROUTER_API_KEY, MODEL_NAME, APP_TITLE

# Initialize the OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "http://localhost:7860",
        "X-Title": APP_TITLE
    }
)

def get_system_prompt() -> str:
    """Reads the system prompt from file, with a fallback if missing."""
    file_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'system.txt')
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "You are Nyahu AI, a highly capable assistant powered by Gemma 4 31B."

def panggil_nyahu_ai(pesan: str, riwayat: list) -> str:
    """
    Main function for Gradio ChatInterface.
    Parses history and calls the OpenRouter API.
    """
    system_prompt = get_system_prompt()

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Parse history
    for item in riwayat:
        # Check if the item is a dictionary (newer Gradio format)
        if isinstance(item, dict):
            # Gradio might use different keys depending on versions.
            # Typical for new format is it has "role" and "content"
            role = item.get("role")
            content = item.get("content")

            # If standard dictionary format is used by gradio dicts
            if role and content:
                messages.append({"role": role, "content": content})

        # Check if item is a list/tuple of 2 items (older Gradio format)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, assistant_msg = item
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})

    # Append the current user message
    messages.append({"role": "user", "content": pesan})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            extra_body={
                "reasoning": {
                    "enabled": True
                }
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while connecting to Nyahu AI: {str(e)}\nPlease try again later."
