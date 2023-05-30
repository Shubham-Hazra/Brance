# Import necessary libraries
import streamlit as st
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
from chatbot import *

# Set Streamlit page configuration
st.set_page_config(page_title='ChatBot🤖', layout='wide')

# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []
if 'input_text' not in st.session_state:
    st.session_state.input_text = ''
user_input = None

def submit():
    st.session_state.input_text = st.session_state.input
    st.session_state.input = ''

# Define function to get user input
def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
    st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Your AI assistant here! Ask me anything ...", 
                            label_visibility='hidden',on_change=submit)
    text = st.session_state.input_text
    if text:
        st.session_state.input_text = ''
        return text 

# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.input_text = ""
    if USE_QUESTION_PROMPT:                                        
        chatbot.make_conversational_chain(k = 6,verbose= True,condense_question_prompt = QUESTION_PROMPT)
    else:
        chatbot.make_conversational_chain(k = 6,verbose= True)

# Set up sidebar with various options
with st.sidebar.expander("🛠️ ", expanded=True):
    COLLECTION_NAME = st.selectbox(label='Collection Name', options=['sbnri_full','sbnri','vetic','blue_nectar'])
    TEMPERATURE = st.slider(label='Temperature', min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    SHOW_SOURCES = st.checkbox("Show Sources", value=False, key="show_sources")
    SYSTEM_PROMPT = st.text_area(label='System Prompt',value="")
    QUESTION_PROMPT = st.text_area(label= 'Question Prompt', value="")
    USE_QUESTION_PROMPT = st.checkbox("Use question prompt?", value=False, key="use_question_prompt")

# Set up the Streamlit app layout
st.title("🤖 Brance Chat Bot")
st.subheader(" Powered by 🦜 LangChain + OpenAI + Streamlit")

# Ask the user to enter their OpenAI API key
API_O = st.sidebar.text_input("API-KEY", type="password")

API_O = os.getenv('OPENAI_API_KEY')

# Session state storage would be ideal
if API_O and COLLECTION_NAME:
    chatbot = Chatbot(temperature = TEMPERATURE)
    chatbot.load_vector_store(COLLECTION_NAME,"src/data/chroma")
    if SYSTEM_PROMPT == "":
        SYSTEM_PROMPT = '''
        You are a customer support guide representing SBNRI, a reputable online platform known for solving the banking needs of NRI's in India. 
        Your role is to assist customers by providing accurate information, offering helpful recommendations, and guiding them towards the solutions of their issues. 
        Feel free to ask clarifying questions only if needed, to better understand the customer's needs and preferences. 
        Leverage the provided context and information in the question itself to answer the question effectively without generating false or fictional information. 
        Double check your response for accuracy. Your responses should be short, friendly and humanlike.
        Respond only to the following question using only the context and the information given in the question.
        Only use your existing knowledge for generic information and not for specific information. Do not make up any figures or facts.
        If you don't know the answer respond with "May I connect you with an expert in this topic to discuss this in detail?":

        Context: 
        {context}

        Question: {question}

        Remember, your expertise and helpfulness are key in assisting customers in making informed choices.'''

    SYSTEM_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT)

    if QUESTION_PROMPT == "":
        QUESTION_PROMPT = """
        The following is a friendly conversation between a human and an AI. Given below is the summary of the conversation between them followed by a question from the human. 
        Append the summary to the question without modifying them. Keep the structure as given below. DO NOT CHANGE ANYTHING.

        Conversation summary:
        {chat_history}

        Question: {question}"""
    
    QUESTION_PROMPT = PromptTemplate.from_template(QUESTION_PROMPT)

    chatbot.set_system_prompt(system_prompt= SYSTEM_PROMPT)  

    if USE_QUESTION_PROMPT:                                        
        chatbot.make_conversational_chain(k = 6,verbose= True,condense_question_prompt = QUESTION_PROMPT)
    else:
        chatbot.make_conversational_chain(k = 6,verbose= True)
else:
    st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')
    st.stop()

# Get the user input
user_input = get_text()

# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
if user_input:
    response = chatbot.chain({'question': user_input})
    source = response["source_documents"]
    urls = ""
    print("\n\nSources:\n")
    for document in source:
        print(f"Url: {document.metadata['source']}")
        urls += document.metadata['source']
        urls += "\n"
    if SHOW_SOURCES:
        output = response["answer"] + "\n" + urls
    else:
        output = response["answer"]
    st.session_state.past.append(user_input)  
    st.session_state.generated.append(output)  


# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")


# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

# Add a button to start a new chat
st.sidebar.button("New Chat", on_click = new_chat, type='primary')

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:   
    st.sidebar.button("Clear-all", on_click= lambda: st.session_state.stored_session.clear(), type='primary')        



