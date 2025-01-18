from run import run_chat, read_config 
import time
import gradio as gr


def slow_echo(assistant, message):
    generated_text = ""
    for i in assistant.message_cicle(message):
      generated_text += i
      
    return generated_text 
      

if __name__ == "__main__":
    assistant = run_chat()
    config = read_config()
    gconfig = config['gradio_config']

    gr.ChatInterface(
        fn=lambda message, history: slow_echo(assistant, message=message),
        type="messages",
        chatbot=gr.Chatbot(height=gconfig['height']),
        textbox=gr.Textbox(
            placeholder=gconfig['textbox_placeholder'], 
            container=gconfig['textbox_container'], 
            scale=gconfig['textbox_scale']
        ),
        title=gconfig['title'],
        description=gconfig['description'],
        theme=gconfig['theme'],
        cache_examples=gconfig['cache_examples'],
    ).launch(share=config['share_gradio'])