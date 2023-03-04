import gradio as gr
import os
import openai
import requests
import json

openai.api_key = os.environ.get("OPENAI_API_KEY")

prompt_templates = {"Default ChatGPT": ""}

def get_empty_state():
    return {"total_tokens": 0, "messages": []}

def download_prompt_templates():
    url = "https://raw.githubusercontent.com/f/awesome-chatgpt-prompts/main/prompts.csv"
    response = requests.get(url)

    for line in response.text.splitlines()[1:]:
        act, prompt = line.split('","')
        prompt_templates[act.replace('"', '')] = prompt.replace('"', '')

    choices = list(prompt_templates.keys())
    return gr.update(value=choices[0], choices=choices)

def on_token_change(user_token):
    openai.api_key = user_token or os.environ.get("OPENAI_API_KEY")

def on_prompt_template_change(prompt_template):
    if not isinstance(prompt_template, str): return
    return prompt_templates[prompt_template]

def submit_message(user_token, prompt, prompt_template, temperature, max_tokens, state):

    history = state['messages']

    if not prompt:
        return gr.update(value='', visible=state['total_tokens'] < 1_000), [(history[i]['content'], history[i+1]['content']) for i in range(0, len(history)-1, 2)], f"Total tokens used: {state['total_tokens']} / 3000", state
    
    prompt_template = prompt_templates[prompt_template]

    system_prompt = []
    if prompt_template:
        system_prompt = [{ "role": "system", "content": prompt_template }]

    prompt_msg = { "role": "user", "content": prompt }
    
    try:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=system_prompt + history + [prompt_msg], temperature=temperature, max_tokens=max_tokens)

        history.append(prompt_msg)
        history.append(completion.choices[0].message.to_dict())

        state['total_tokens'] += completion['usage']['total_tokens']
    
    except Exception as e:
        history.append(prompt_msg)
        history.append({
            "role": "system",
            "content": f"Error: {e}"
        })

    total_tokens_used_msg = f"Total tokens used: {state['total_tokens']} / 3000" if not user_token else ""
    chat_messages = [(history[i]['content'], history[i+1]['content']) for i in range(0, len(history)-1, 2)]
    input_visibility = user_token or state['total_tokens'] < 3000

    return gr.update(value='', visible=input_visibility), chat_messages, total_tokens_used_msg, state

def clear_conversation():
    return gr.update(value=None, visible=True), None, "", get_empty_state()

css = """
      #col-container {max-width: 80%; margin-left: auto; margin-right: auto;}
      #chatbox {min-height: 400px;}
      #header {text-align: center;}
      #prompt_template_preview {padding: 1em; border-width: 1px; border-style: solid; border-color: #e0e0e0; border-radius: 4px;}
      #total_tokens_str {text-align: right; font-size: 0.8em; color: #666; height: 1em;}
      #label {font-size: 0.8em; padding: 0.5em; margin: 0;}
      .message { font-size: 1.2em; }
      """

with gr.Blocks(css=css) as demo:
    
    state = gr.State(get_empty_state())


    with gr.Column(elem_id="col-container"):
        gr.Markdown("""## Sydne-AI
                    Current limit is 3000 tokens per conversation.""",
                    elem_id="header")

        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(elem_id="chatbox")
                input_message = gr.Textbox(show_label=False, placeholder="Enter text and press submit", visible=True).style(container=False)
                btn_submit = gr.Button("Submit")
                total_tokens_str = gr.Markdown(elem_id="total_tokens_str")
                btn_clear_conversation = gr.Button("Restart Conversation")
            with gr.Column():
                prompt_template = gr.Dropdown(label="Set a custom insruction for the chatbot:", choices=list(prompt_templates.keys()))
                prompt_template_preview = gr.Markdown(elem_id="prompt_template_preview")
                gr.Markdown("Enter your own OpenAI API Key to remove the 3000 token limit. You can get it [here](https://platform.openai.com/account/api-keys).", elem_id="label")
                user_token = gr.Textbox(placeholder="OpenAI API Key", type="password", show_label=False)
                with gr.Accordion("Advanced parameters", open=False):
                    temperature = gr.Slider(minimum=0, maximum=2.0, value=0.7, step=0.1, interactive=True, label="Temperature (higher = more creative/chaotic)")
                    max_tokens = gr.Slider(minimum=100, maximum=4096, value=1000, step=1, interactive=True, label="Max tokens per response")

    # gr.HTML('''<br><br><br><center><a href="https://huggingface.co/spaces/anzorq/chatgpt-demo?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>You can duplicate this Space.<br>
    #         Don't forget to set your own <a href="https://platform.openai.com/account/api-keys">OpenAI API Key</a> environment variable in Settings.<br>
    #         <p><img src="https://visitor-badge.glitch.me/badge?page_id=anzorq.chatgpt_api_demo_hf" alt="visitors"></p></center>''')

    btn_submit.click(submit_message, [user_token, input_message, prompt_template, temperature, max_tokens, state], [input_message, chatbot, total_tokens_str, state])
    input_message.submit(submit_message, [user_token, input_message, prompt_template, temperature, max_tokens, state], [input_message, chatbot, total_tokens_str, state])
    btn_clear_conversation.click(clear_conversation, [], [input_message, chatbot, total_tokens_str, state])
    prompt_template.change(on_prompt_template_change, inputs=[prompt_template], outputs=[prompt_template_preview])
    user_token.change(on_token_change, inputs=[user_token], outputs=[])

    
    demo.load(download_prompt_templates, inputs=None, outputs=[prompt_template])


demo.launch(debug=True, height='800px')
