import requests
import gradio as gr
import datetime
import os

HF_API_KEY = os.environ["HF_API_KEY"]
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

CHAT_DIR = "chats"
os.makedirs(CHAT_DIR, exist_ok=True)

def query(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    try:
        return response.json()
    except Exception:
        return [{"generated_text": "[ERROR] Invalid response from model"}]

def save_chat_to_file(history, filename):
    with open(os.path.join(CHAT_DIR, filename), "w", encoding="utf-8") as f:
        for msg in history:
            role = msg["role"].title()
            content = msg["content"]
            f.write(f"{role}: {content}\n")

def chat_with_model(message, history):
    full_prompt = ""
    for msg in history:
        full_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    full_prompt += f"User: {message}\nAssistant:"

    payload = {
        "inputs": full_prompt,
        "parameters": {"max_new_tokens": 200, "temperature": 0.7}
    }

    response = query(payload)
    if isinstance(response, list) and 'generated_text' in response[0]:
        reply = response[0]["generated_text"].split("Assistant:")[-1].strip()
    else:
        reply = "[ERROR] Invalid model output"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return history, history, ""

def new_chat():
    filename = f"chat_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    return [], [], filename

gr.Chatbot.postprocess = lambda self, x: x
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div style='text-align: center; padding: 1rem;'>
        <h1 style='color: #007BFF;'>Welcome to <span style='color: orange;'>M.O.T</span> — Powered by Mixtral AI</h1>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            new_chat_btn = gr.Button("🆕 New Chat")
            save_btn = gr.Button("💾 Save Chat")
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="M.O.T", type="messages", height=500)
            state = gr.State([])
            chat_file = gr.State("chat_temp.txt")

            with gr.Row():
                txt = gr.Textbox(placeholder="Type your message here...", show_label=False, scale=8)
                send_btn = gr.Button("Send", scale=1)

    txt.submit(chat_with_model, [txt, state], [chatbot, state, txt])
    send_btn.click(chat_with_model, [txt, state], [chatbot, state, txt])
    save_btn.click(lambda history, file: save_chat_to_file(history, file), [state, chat_file], None)
    new_chat_btn.click(new_chat, outputs=[chatbot, state, chat_file])

demo.launch(share=True)
