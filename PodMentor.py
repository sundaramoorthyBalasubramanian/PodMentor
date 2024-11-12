import os
import streamlit as st
import whisper
import shutil
from glob import glob
import time
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
#import pinecone
from dotenv import load_dotenv
import hashlib
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
#from langchain_community.vectorstores import Pinecone
from langchain.document_loaders import DirectoryLoader
import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import TextLoader
import langchain_community

print("Start Title")
st.title("PodMentor – Interactive knowledge from your podcasts & docs. ")

upload_directory = 'tmp'
ALREADY_UPLOADED = "Already Uploaded"
UPLOAD_NEW = "Upload new one"
#tmp_directory = 'tmp'
HUGGINGFACE_API_KEY = "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"
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#content_type =ALREADY_UPLOADED
#model:whisper.model.Whisper = None
#llm :langchain_community.llms.huggingface_hub.HuggingFaceHub = None

@st.cache_resource 
def load_models():
    print("Start loading models",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    model = whisper.load_model("turbo")
    print(type(model))
    print("End loading models",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))    
    return model
    
def load_docs(directory: str):
    """
    Load documents from the given directory.
    """
    print("load_docs start:",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    loader = DirectoryLoader(directory,use_multithreading=True)
    documents = loader.load()
    print("load_docs end:",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    return documents

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    """
    Split the documents into chunks.
    """
    print("split_docs start:",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    print("split_docs end:",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    return docs

@st.cache_resource    
def create_LLM():
    HUGGINGFACE_API_KEY = "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"
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    print("create_LLM start:",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50}, huggingfacehub_api_token=HUGGINGFACE_API_KEY
    )
    print("create_LLM end:",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    return llm

@st.cache_resource    
def loadWhisperTurboModel():
    print("loadWhisperTurboModel start:",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    #os.environ["TRANSFORMERS_CACHE"] = "C:\\Users\\balasubramanians\\.cache\\whisper"

    whisperSModel = whisper.load_model("turbo", download_root="C:\\Users\\balasubramanians\\.cache\\whisper")
    print("loadWhisperTurboModel end:",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    return whisperSModel
    
def startup_event():
    """
    Load all the necessary models and data once the server starts.
    """
    documents = load_docs(upload_directory+"/")
    docs = split_docs(documents)
    print("SentenceTransformerEmbeddings start:",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    print("SentenceTransformerEmbeddings end:",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    persist_directory = "chroma_db"
    print("vectordb start:",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print("vectordb end:",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    
    llm = create_LLM()
    
    print("load_qa_chain start:",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
    print("load_qa_chain end:",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

    return vectordb,chain
    
def get_answer(query: str, db, chain):
    """
    Queries the model with a given question and returns the answer.
    """
    print("get_answer start:",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    matching_docs_score = db.similarity_search_with_score(query)

    matching_docs = [doc for doc, score in matching_docs_score]
    answer = chain.run(input_documents=matching_docs, question=query)

    # Prepare the sources
    sources = [{
        "content": doc.page_content,
        "metadata": doc.metadata,
        "score": score
    } for doc, score in matching_docs_score]
    
    print("get_answer end:",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    return {"answer": answer, "sources": sources}

@st.fragment
def QA(db, chain):
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            print("markdown user")
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = get_answer(st.session_state.messages[-1]["content"], db, chain)
            answer = full_response['answer']
            message_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
   
def start_chatbot():
    if os.path.exists(upload_directory):
        print("calling startup_event")
        db, chain = startup_event()
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            print("markdown role")
            st.markdown(message["content"])
    
    QA(db,chain)

def callback(content_type:str):
    st.session_state.button_clicked = True
    curr_dir=""
    model = load_models()    
    if content_type == UPLOAD_NEW:
        print("Display uploader UI")
        uploaded_files = st.sidebar.file_uploader("Choose file/folder contains .docx/.pdf/.txt/.mp3/.mp4", accept_multiple_files=True)
       
        if uploaded_files is not None and len(uploaded_files):
            if os.path.exists(upload_directory):
                shutil.rmtree(upload_directory)
            
            print("Create Upload directory")        
            os.makedirs(upload_directory)
            
            for file in uploaded_files:
                file_path = upload_directory+"/"+file.name
                fileExt = file.name[-4:]
                print("This is Media file file_path1,filename,fileExt",file_path,file.name,fileExt)
                print("step4")
                with open(file_path, 'wb') as temp:
                    if fileExt == ".mp3" or fileExt ==".mp4":
                        print("transcribe start:",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                        transcription = model.transcribe(file.name)
                        print("transcribe end:",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                        filename = file.name.replace(fileExt,'')
                        filename = f"{upload_directory}/{filename}"+'.txt'
                        print("This is Media file filename:"+filename)
                        with open(filename, 'w') as output:
                            output.write(transcription['text'])
                    else:        
                        temp.write(file.getvalue())
                        temp.seek(0)
                    
                curr_dir = [path.split(os.path.sep)[-1] for path in glob(upload_directory + '/*')]
                print("curr_dir:",curr_dir)
                    

    elif content_type == ALREADY_UPLOADED:
        st.sidebar.write("Current Knowledge Base")
        if len(curr_dir):
            st.sidebar.write(curr_dir)
        else:
            st.sidebar.write('**No KB Uploaded**')
          
    if curr_dir and len(curr_dir):
        print("calling start_chatbot") 
        start_chatbot()
    else:
        st.header('No KB Loaded, use the left menu to start')    
        

content_type = st.sidebar.radio("Which Knowledge base you want to use?",
                                [ALREADY_UPLOADED, UPLOAD_NEW])

print("callback enter content_type:",content_type)
callback(content_type)


