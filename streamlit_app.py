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

# Define function to get user input
def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Your AI assistant here! Ask me anything ...", 
                            label_visibility='hidden')
    return input_text

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
    chatbot.make_conversational_chain()

# Set up sidebar with various options
with st.sidebar.expander("🛠️ ", expanded=False):
    COLLECTION_NAME = st.selectbox(label='Collection Name', options=['sbnri_full','sbnri','vetic','blue_nectar'])
    TEMPERATURE = st.slider(label='Temperature', min_value=0.0, max_value=1.0, value=0.0, step=0.1)

# Set up the Streamlit app layout
st.title("🤖 Brance Chat Bot")
st.subheader(" Powered by 🦜 LangChain + OpenAI + Streamlit")

# Ask the user to enter their OpenAI API key
# API_O = st.sidebar.text_input("API-KEY", type="password")

API_O = os.getenv('OPENAI_API_KEY')

# Session state storage would be ideal
if API_O and COLLECTION_NAME:
    chatbot = Chatbot(temperature = TEMPERATURE)
    chatbot.load_vector_store(COLLECTION_NAME,"src/data/chroma")
    chatbot.make_conversational_chain()
else:
    pass
    # st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')
    # st.stop()


# Add a button to start a new chat
st.sidebar.button("New Chat", on_click = new_chat, type='primary')

# Get the user input
user_input = get_text()

# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
if user_input:
    output = chatbot.chain({'question': user_input})["answer"]
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

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:   
    st.sidebar.button("Clear-all", on_click= lambda: st.session_state.stored_session.clear(), type='primary')        



