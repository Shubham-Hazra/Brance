import os
import time

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
import gradio as gr
import qdrant_client
import requests
import xmltodict
from bs4 import BeautifulSoup
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import (ConversationBufferMemory,
                              ConversationSummaryBufferMemory)
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Qdrant

HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'}

def get_open_ai_model(model_name = 'gpt-3.5-turbo',temperature = 0, openai_api_key = os.getenv("OPENAI_API_KEY"),chat_model = True, verbose = False):
    if chat_model:
        model =  ChatOpenAI(model_name=model_name,temperature=temperature,openai_api_key=openai_api_key,verbose=verbose)
    else:
        model =  OpenAI(model_name=model_name,temperature=temperature,openai_api_key=openai_api_key,verbose=verbose)

    return model

def extract_text_from_url(url, headers = None):
            if headers is None:
                headers = HEADERS
            html = requests.get(url,headers=headers).text
            soup = BeautifulSoup(html, features="html.parser")
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            return '\n'.join(line for line in lines if line)

def get_urls(sitemap_url, headers = None):
            urls = []
            if headers is None:
                headers = HEADERS
            sitemap = requests.get(sitemap_url,headers=headers).text
            try:
                sitemap = xmltodict.parse(sitemap)
                if 'sitemapindex' in sitemap:
                    sitemap = sitemap['sitemapindex']['sitemap']
                    for entry in sitemap:
                        urls += get_urls(entry['loc'])
                else:
                    sitemap = sitemap['urlset']['url']
                    for entry in sitemap:
                        urls.append(entry['loc'])
            except:
                print(f"Error parsing sitemap {sitemap}")
            return urls

def get_pages(urls):
            pages = []
            for url in urls:
                try:
                    pages.append({'text': extract_text_from_url(url), 'source': url})
                except Exception as e:
                    print(e)
            return pages

def get_document_chunks(pages,verbose = False, text_splitter = None):
    if text_splitter is None:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],chunk_overlap=200,)
    doc_chunks = []
    for page in pages:
        chunks = text_splitter.split_text(page['text'])
        for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={"source": page['source']},
                )
                doc_chunks.append(doc)
        if verbose:
            print(f"Split {page['source']} into {len(chunks)} chunks")
    return doc_chunks


def load_vector_store(collection_name,persist_directory,embedding = None , type = 'chroma'):
    if embedding is None:
        embedding = OpenAIEmbeddings()

    if type == 'chroma':
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
        )

    if type == 'qdrant':
        client = qdrant_client.QdrantClient(
            path=persist_directory, prefer_grpc=True
        )
        vector_store = Qdrant(
            client=client, collection_name=collection_name, 
            embedding=embedding
        )

    return vector_store

def get_vector_store(doc_chunks,collection_name,persist_directory, embedding = None , type = 'chroma'):
    if embedding is None: embedding = OpenAIEmbeddings()

    if type == 'chroma':
        vector_store = Chroma.from_documents(
        doc_chunks,
        collection_name= collection_name,
        persist_directory = persist_directory,
        embedding= embedding
        )
        
        vector_store.persist()


    if type == 'qdrant':
        vector_store = Qdrant.from_documents(
            doc_chunks,
            collection_name= collection_name,
            path = persist_directory,
            embedding= embedding
        )

    return vector_store


def get_conversational_chain(vector_store,model = None,memory = None , return_source_documents = True , verbose = False, system_prompt = None, search_type = 'mmr', k = 2, **kwargs):
    if model is None:
        model = get_open_ai_model()

    if memory is None:    
        memory = ConversationSummaryBufferMemory(llm=model , memory_key="chat_history", return_messages=True,input_key='question', output_key='answer')

    if system_prompt is None:
        chain = ConversationalRetrievalChain.from_llm(
        model, 
        vector_store.as_retriever(search_type = search_type, k = k), 
        memory=memory,
        return_source_documents=return_source_documents,
        verbose=verbose,
        **kwargs
        )
    else:
        chain = ConversationalRetrievalChain.from_llm(
        model, 
        vector_store.as_retriever(search_type = search_type, k = k), 
        memory=memory,
        return_source_documents=return_source_documents,
        combine_docs_chain_kwargs=dict(prompt=system_prompt),
        verbose=verbose,
        **kwargs
        )

    return chain

def launch(chain,share = False):
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")            
        def user(user_message, history):
            return "", history + [[user_message, None]]
        def bot(history):
            response = chain({"question": history[-1][0]})
            source = response["source_documents"]
            urls = ""
            print("\n\nSources:\n")
            for document in source:
                print(f"Url: {document.metadata['source']}")
                urls += document.metadata['source']
                urls += "\n"
            bot_message = response["answer"] + "\n" + urls
            history[-1][1] = ""
            for character in bot_message:
                history[-1][1] += character
                time.sleep(0.01)
                yield history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: chain.memory.clear() ,  None, chatbot, queue=False)
    demo.queue()
    demo.launch(share=share)

