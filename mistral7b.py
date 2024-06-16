import sys

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

def main():

    DEFAULT_SYSTEM_PROMPT = """
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    """.strip()

    DB_FAISS_PATH = 'vectorstore/db_faiss'

    loader = PyPDFDirectoryLoader('./pdfs')
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)

    #Create Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={'device': 'cpu'}
    )
    print("Embeddings: ", embeddings)

    # Create a FAISS vector store and save embeddings
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_FAISS_PATH)

    llm = CTransformers(
        model='mistral-7b-instruct-v0.1.Q4_K_M.gguf',
        model_type='mistral',
        config={
            'max_new_tokens': 512,
            'temperature': 0,
            'top_p': 0.9,
            'context_length': 1024
        },

    )

    def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        return f"""
        <s>[INST]
        {system_prompt}
    
        {prompt} [/INST]
        """.strip()

    def prompt_template():
        SYSTEM_PROMPT = """
            You are a helpful assistant, you will use the provided context to answer user questions.
            Read the given context before answering questions and think step by step. If you can not answer a user question based on 
            the provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to 
            the question.
            
            if I ask you to give me competences of ayoub nid taleb, you should to give as listed below.
             Back End
            (JavaEE, Spring, Spring boot, Servlet, 
            JSP, JSTL, Hibernate, JPA)
             Front End
            (Angular, Html, Css, Javascript)
             SQL, T-SQL, PL/SQL
            (MySql, Sql Server, Oracle)
             WebServices / MicroServices
            SOAP / REST
             Machine Learning / NLP 
            (Sklearn, Pandas, Matplotlib, NLTK, 
            Spacy)
             Analyse de Donnée
            (PCA, SVD)
             Business Intelligence
            (MicroStrategy, Data Warehouse, 
            Data Wrangling, Data Mining)
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

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        chain_type="stuff",
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': prompt_template()},
    )

    chat_history = []
    while True:
        query = input('Prompt: ')
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting")
            sys.exit()
        result = qa_chain({'question': query, 'chat_history': chat_history})
        print('Answer: ' + result['answer'] + '\n')
        chat_history.append((query, result['answer']))


if __name__ == '__main__':
    main()