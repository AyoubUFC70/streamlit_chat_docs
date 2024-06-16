import base64
import os.path
import tempfile

import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from template import css, bot, user

DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()

DB_FAISS_PATH = 'vectorstore/db_faiss'


def init_page():
    load_dotenv()
    st.set_page_config(page_title="🦙💬 Mistral-7b Chatbot")

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello, I am an assistant, ask me about your documents"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hi !"]


    # st.write(css, unsafe_allow_html=True)
    #
    # if "conversation" not in st.session_state:
    #     st.session_state.conversation = None
    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = None

    st.header("Chat with multiple PDFs using 🦙💬 Llama2-7b Chatbot")

    # with open('images/eviden.png', "rb") as img_file:
    #     encoded_string = base64.b64encode(img_file.read())
    # st.markdown(
    #     f"""
    #     <style>
    #     .stApp {{
    #         background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
    #         background-size: cover;
    #     }}
    #     .st-emotion-cache-19rxjzo {{
    #         background: transparent;
    #     }}
    #     .st-emotion-cache-19rxjzo:hover {{
    #         color: #fff;
    #     }}
    #     .st-emotion-cache-0 {{
    #         background: transparent;
    #     }}
    #     .st-bq st-ca st-cb st-cc st-cd st-ce st-cf st-cg st-ch st-ci st-cj st-ae st-ck st-cl st-cm st-cn st-co st-cp st-cq st-cr st-ay st-az st-b0 st-cs st-b2 st-b3 st-c9 st-ct st-cu st-cv {{
    #         background: transparent;
    #         color: #fff;
    #     }}
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )
    with open('images/eviden-logo.png', "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read())
    st.markdown(
        f"""
            <style>
                [data-testid="stSidebarNav"] {{
                    background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
                    padding-top: 120px;
                    background-position: 20px 20px;
                }}
            </style>
        """,
        unsafe_allow_html=True
    )

    # user_input = st.text_input("Ask questions about your documents")
    # if user_input:
    #     handle_user_input(user_input)

    # st.write(user.replace("{{MSG}}", "Hello Robot"), unsafe_allow_html=True)
    # st.write(bot.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)

def load_pdf(pdf_files):
    texts = []
    for pdf in pdf_files:
        pdf_extension = os.path.splitext(pdf.name)[1]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf.read())
            temp_file_path = temp_file.name

        loader = None
        if pdf_extension == '.pdf':
                loader = PyPDFLoader(temp_file_path)

        if loader:
            texts.extend(loader.load())
            os.remove(temp_file_path)

    return texts

def split_to_chunks(texts):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(texts)
    print("Text chunks: ", chunks)
    return chunks

#Create Embeddings
def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={'device': 'cpu'}
    )
    print("Embeddings: ", embeddings)
    return embeddings

# Create Vector Store
def create_vectorstore(text_chunks):
    embeddings = create_embeddings()
    # Create a FAISS vector store and save embeddings
    db = FAISS.from_documents(text_chunks, embedding=embeddings)
    db.save_local(DB_FAISS_PATH)
    return db

def create_llm(config):
    # config = { "temperature": 0.01,
    #            "max_new_tokens": 512,
    #            "repetition_penalty": 1.15,
    #            "top_p": 0.95,
    #            "context_length": 2048
    #         }

    llm = CTransformers(
        model='mistral-7b-instruct-v0.1.Q4_K_M.gguf',
        model_type='mistral',
        config=config
    )
    return llm

def instantiate_memory():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return memory

def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    return f"""
    [INST] <<SYS>>
    {system_prompt}
    <<SYS>>

    {prompt} [/INST]
    """.strip()

def prompt_template():
    SYSTEM_PROMPT = """
        You are a helpful assistant, you will use the provided context to answer user questions.
        Read the given context before answering questions and think step by step. If you can not answer a user question based on 
        the provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to the question.

    """
    template = generate_prompt(
        """
    Context: {context}

    Question: {question}
    """,
        system_prompt=SYSTEM_PROMPT,
    )
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    return prompt


def chain_qa_retrieval(llm, vectorstore):
    try:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff",
            memory=instantiate_memory(),
            #return_source_documents=True,
            #chain_type_kwargs={'prompt': prompt_template()}
            combine_docs_chain_kwargs={'prompt': prompt_template()},
            verbose=True,
        )
        return qa_chain
    except Exception as e:
        print("Error creating qa_chain : ", e)
        return None

def handle_user_input(query):
    response = st.session_state.conversation({'question': query})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def clear_chat_history():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you today?"}]
    instantiate_memory().clear()

def main():
    init_page()

    st.sidebar.title("Upload Your Documents 📁")
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.0, step=0.01)
    top_p = st.sidebar.slider('top_q', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    repetition_penalty = st.sidebar.slider('repetition_penalty', min_value=0.1, max_value=2.0, value=1.15, step=0.01)
    max_tokens = st.sidebar.slider('max_tokens', min_value=10.0, max_value=1024.0, value=1024.0, step=10.0)


    config = {"temperature": temperature,
              "max_new_tokens": max_tokens,
              "repetition_penalty": repetition_penalty,
              "top_p": top_p,
              "context_length": 4096
              }

    pdf_files = st.sidebar.file_uploader("You can Upload your PDFs here", accept_multiple_files=True)
    if pdf_files:
        # Load pdf files
        texts = load_pdf(pdf_files)

        # Split data into chunks
        chunk = split_to_chunks(texts)

        # Create Vector Store
        vectorstore = create_vectorstore(chunk)

        llm = create_llm(config)

        chain_qa = chain_qa_retrieval(llm, vectorstore)

        st.sidebar.success("File Loaded Successfully!!!")

        def conversational_chat(query):
            response = chain_qa({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, response["answer"]))
            return response["answer"]

        response = st.container()
        user_input = st.container()

        with user_input:
            with st.form(key='my_form', clear_on_submit=True):
                user_question = st.text_input("Prompt:", placeholder="Ask me about your documents", key='input')
                submit_btn = st.form_submit_button(label='Send')

            if submit_btn and user_question:
                with st.spinner("Generating response..."):
                    result = conversational_chat(user_question)
                    st.session_state['past'].append(user_question)
                    st.session_state['generated'].append(result)

        if st.session_state['generated']:
            with response:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style='big-smile')
                    message(st.session_state['generated'][i], key=str(i), avatar_style='')

    # user_input = st.text_input("Ask questions about your documents")
    # if user_input:
    #     response = chain_qa({"question": user_input, "chat_history": st.session_state['history']})
    #     st.session_state['history'].append((user_input, response["answer"]))
    #     st.write(response)

                # Create Converation Chain
                # st.session_state.conversation = chain_qa_retrieval(vectorstore)



if __name__ == '__main__':
    main()