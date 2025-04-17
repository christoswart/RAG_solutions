import os
import glob
from dotenv import load_dotenv
import gradio as gr

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

MODEL = "gpt-4o-mini"
db_name = "vector_db"

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

# Read in documents using LangChain's loaders
# Take everything in all the sub-folders of our knowledgebase
folders = glob.glob("knowledge-base/*")

text_loader_kwargs = {'encoding': 'utf-8'}
# If that doesn't work, some Windows users might need to uncomment the next line instead
# text_loader_kwargs={'autodetect_encoding': True}

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

print(f"Number of documents: ", len(documents))

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(f"Number of chunks: ", len(chunks))
print(f"First chunk: ", chunks[0].page_content[:100])

doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
print(f"Document types found: {', '.join(doc_types)}")

# Using OpenAI Subscription: Put the chunks of data into a Vector Store that associates a Vector Embedding with each chunk
# embeddings = OpenAIEmbeddings()

# If you would rather use the free Vector Embeddings from HuggingFace sentence-transformers, you can uncomment the next two lines and comment out the OpenAIEmbeddings line above.
from langchain_huggingface import HuggingFaceEmbeddings 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Check if a Chroma Datastore already exists - if so, delete the collection to start from scratch
#if os.path.exists(db_name):
    #Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
#vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)


# Create our FAISS vectorstore!
# FAISS is a library for efficient similarity search and clustering of dense vectors
# It is a great choice for large-scale vector search and is used by many companies and research projects.
# It is also the default vector store used by LangChain.

# PERSISTED
# You can persist the vectorstore to disk by uncommenting the persist_directory lines below
# If you do this, you will need to delete the vectorstore before re-running this code, or it will throw an error.

#if os.path.exists(db_name):
    #FAISS(persist_directory=db_name, embedding_function=embeddings).delete_collection()
#vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)

# IN MEMORY
# If you want to use the FAISS vectorstore without persisting it to disk, you can uncomment the lines below and comment out the lines above
# This will create a new vectorstore in memory that will be lost when the program ends.
vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)


total_vectors = vectorstore.index.ntotal
dimensions = vectorstore.index.d
print(f"There are {total_vectors} vectors with {dimensions:,} dimensions in the vector store")


# ~~~ Visualizing the Vector Store ~~~
# We will use t-SNE to reduce the dimensionality of the vectors to 2D for visualization

# setup
# Prework
vectors = []
documents = []
doc_types = []
colors = []
color_map = {'products':'blue', 'employees':'green', 'contracts':'red', 'company':'orange'}

for i in range(total_vectors):
    vectors.append(vectorstore.index.reconstruct(i))
    doc_id = vectorstore.index_to_docstore_id[i]
    document = vectorstore.docstore.search(doc_id)
    documents.append(document.page_content)
    doc_type = document.metadata['doc_type']
    doc_types.append(doc_type)
    colors.append(color_map[doc_type])
    
vectors = np.array(vectors)

# We humans find it easier to visalize things in 2D!
# Reduce the dimensionality of the vectors to 2D using t-SNE
# (t-distributed stochastic neighbor embedding)

tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 2D scatter plot
fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='2D FAISS Vector Store Visualization',
    scene=dict(xaxis_title='x',yaxis_title='y'),
    width=800,
    height=600,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()


# Let's try 3D Visualization!
tsne = TSNE(n_components=3, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='3D FAISS Vector Store Visualization',
    scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
    width=900,
    height=700,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()


# create a new Chat with OpenAI
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever()

# putting it together: set up the conversation chain with the GPT 3.5 LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

query = "Can you describe Insurellm in a few sentences"
result = conversation_chain.invoke({"question":query})
print(result["answer"])

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]

view = gr.ChatInterface(chat).launch(inbrowser=True, share=True, debug=True)