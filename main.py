import os
import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- CONFIG ----------------
load_dotenv()
api_key = os.getenv('OPENAI_KEY')

embedding_function = OpenAIEmbeddings(
    openai_api_key=api_key,
    model='text-embedding-3-small'
)

rc_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=200,
    chunk_overlap=100
)

# ---------------- FUNCTIONS ----------------
def document_splitter(document):
    return rc_splitter.split_documents(document)

def create_collection(name, docs):
    labeled_docs = []

    for i, doc_text in enumerate(docs):
        labeled_docs.append(
            Document(
                page_content=str(doc_text),  # force string safety
                metadata={"paperNo": f"paper{i+1}"}
            )
        )

    return Chroma.from_documents(
        documents=labeled_docs,
        embedding=embedding_function,
        persist_directory="./chroma_db",
        collection_name=name
    )


#persistent
#client = chromadb.PersistentClient(path="./chroma_db")

client  = chromadb.EphemeralClient()

def delete_collections():
    for col in client.list_collections():
        client.delete_collection(name=col.name)
    print("All collections deleted ")

def list_collections():
    print(client.list_collections())


def load_documents(paths: list[str]):
    docs = []
    for path in paths:
        loader = PyPDFLoader(path)
        document = loader.load()
        docs_splitted = document_splitter(document)
        docs.extend(docs_splitted)
    return docs

def add_to_collection(collection, document):
    pass


# ---------------- PIPELINE ----------------
# file_paths = ["documents/airline_policy.pdf",
#               "documents/CV-ubongudoette_AI5.pdf",
#               "documents/term_semester_dates_herts.pdf",
#               "documents/you_only_look_once_unified_real_time_object_detection.pdf"
#               ]


#file_path = ["documents/you_only_look_once_unified_real_time_object_detection.pdf"]
#docs = load_documents(file_paths)

#
# print("Process setup completed successfully ")
# print(list_collections())
# #delete_collections()
