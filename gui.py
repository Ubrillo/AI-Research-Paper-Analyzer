import os
import main
import streamlit as st
from chatbot import ChatBot
from main import load_documents, create_collection

st.title("AI Research Chatbot")
question = st.text_input("Question: ")

upload_dir = "uploaded_docs"
os.makedirs(upload_dir, exist_ok=True)


docs = []

st.sidebar.header("Upload your documents")

for i in range(3):
    st.sidebar.markdown(f"### Document Slot {i+1}")


    #File uploadr per slot
    upload_file = st.sidebar.file_uploader(
        f"Upload your file {i+1}",
        type=["pdf"],
        key=f"file_{i+1}",
    )

    if upload_file is not None:
        save_path = os.path.join(upload_dir, upload_file.name)

        #save file locally
        with open(save_path, "wb") as f:
            f.write(upload_file.getbuffer())

        st.sidebar.success(f"Uploaded: {upload_file.name}")
        docs.append(save_path)
    else:
        docs.append(None)

processed_doc_clicked = st.sidebar.button("Load documents")
main_placeholder = st.empty()

if processed_doc_clicked:
    valid_docs = [doc for doc in docs if doc != None]
    main_placeholder.text("Loading documents...started...📎")

    docs = load_documents(valid_docs)
    create_collection('research_lake', docs)

    main_placeholder.text("Documents loaded into vector DB ✅")

chatbot = ChatBot()
if question:
    answer = chatbot.ask(question.lower())
    st.header("Answer:")
    st.write(answer)