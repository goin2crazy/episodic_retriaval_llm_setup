from run import run_chat
import time
import gradio as gr


def slow_echo(assistant, message):
    generated_text = ""
    for i in assistant.message_cicle(message):
      generated_text += i
      yield generated_text 
      

if __name__ == "__main__": 
    assistant = run_chat() 

    gr.ChatInterface(
        fn=lambda message, history: slow_echo(assistant, message=message), 
        type="messages"
    ).launch()