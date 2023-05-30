from brance import *


class Chatbot():
    def __init__(self, model = None, temperature = 0):
        if model is None: self.model = get_open_ai_model(temperature =temperature)
        
        self.memory = ConversationSummaryBufferMemory(llm=self.model , memory_key="chat_history", return_messages=True,input_key='question', output_key='answer')
        
        self.system_prompt = PromptTemplate.from_template('''
        # Respond to the following question using only the context, and if you don't know the answer, 
        # simply respond accordingly without making up information:

        # Context: {context}
        # Question: {question}

        # You can also ask questions to gather more insights and guide the user more effectively. Remember, your expertise and helpfulness are crucial in assisting users 
        # with their technical concerns and providing them with the best solutions.'''
        )
    
    def set_model(self,model):
        self.model = model

    def load_vector_store(self,collection_name,persist_directory,embedding = None , type = 'chroma'):
        self.vector_store = load_vector_store(collection_name,persist_directory,embedding = embedding , type = type)

    def load_vector_store_from_url(self,url, collection_name,persist_directory,embedding = OpenAIEmbeddings(),verbose=False,headers = None,type = 'chroma'):
        if url[-1] == "/":
            url = url[:-1]
        sitemap_url = url + "/sitemap.xml"
        if verbose:
            print(f"Getting sitemap from {sitemap_url}")
        urls = get_urls(sitemap_url)
        if verbose:
            print(f"Found {len(urls)} urls")
        pages = get_pages(urls)
        doc_chunks = get_document_chunks(pages,verbose)
        if verbose:
            print(f"Created {len(doc_chunks)} document chunks")

        self.vector_store = get_vector_store(doc_chunks,collection_name= collection_name,persist_directory = persist_directory,embedding= embedding,type=type)
    
    def set_memory(self,memory):
        self.memory = memory

    def set_system_prompt(self,system_prompt,**kwargs):
        self.system_prompt = system_prompt
        self.chain = get_conversational_chain(self.vector_store,self.model,self.memory ,system_prompt = system_prompt, **kwargs)
    
    def make_conversational_chain(self,vector_store = None ,model = None,memory = None , system_prompt = None,return_source_documents = True , verbose = False, search_type = 'mmr', k = 2, **kwargs):
        if vector_store is None: vector_store = self.vector_store
        if model is None: model = self.model
        if memory is None: memory = self.memory
        memory = memory.clear()
        if system_prompt is None: system_prompt = self.system_prompt
        self.chain = get_conversational_chain(vector_store,model = model,memory = memory , return_source_documents = return_source_documents , verbose = verbose, system_prompt = system_prompt, search_type = search_type, k = k, **kwargs)

    def launch(self,share = False):
        launch(self.chain,share = share)