import os
import glob
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI

MODEL = "gpt-4o-mini"

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
openai = OpenAI()

context = {}

employees = glob.glob("knowledge-base/employees/*")

for employee in employees:
    name = employee.split(' ')[-1][:-3]
    doc = ""
    with open(employee, "r", encoding="utf-8") as f:
        doc = f.read()
    context[name]=doc

print("Context loaded")
#print(context["Lancaster"])

products = glob.glob("knowledge-base/products/*")

for product in products:
    name = product.split(os.sep)[-1][:-3]
    doc = ""
    with open(product, "r", encoding="utf-8") as f:
        doc = f.read()
    context[name]=doc

print(context.keys())

system_message = "You are an expert in answering accurate questions about Insurellm, the Insurance Tech company. Give brief, accurate answers. If you don't know the answer, say so. Do not make anything up if you haven't been provided with relevant context."

def get_relevant_context(message):
    relevant_context = []
    for context_title, context_details in context.items():
        if context_title.lower() in message.lower():
            relevant_context.append(context_details)
    return relevant_context          

#print(get_relevant_context("Who is lancaster?"))

def add_context(message):
    relevant_context = get_relevant_context(message)
    if relevant_context:
        message += "\n\nThe following additional context might be relevant in answering this question:\n\n"
        for relevant in relevant_context:
            message += relevant + "\n\n"
    return message

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history
    message = add_context(message)
    messages.append({"role": "user", "content": message})

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True, share=True, debug=True)