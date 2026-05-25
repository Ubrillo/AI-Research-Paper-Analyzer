import os
import warnings
from dotenv import load_dotenv
from operator import itemgetter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


class ChatBot:
    def __init__(
        self, 
        db_path: str = "./chroma_db",
        collection_name: str = "research_lake", 
        model: str = "gpt-4o-mini"
        ):
        
        load_dotenv()

        api_key = os.getenv("OPENAI_KEY")
        if not api_key:
            raise ValueError("OPENAI_KEY not found in environment variables.")

        self.api_key = api_key
        self.db_path = db_path
        self.collection_name = collection_name

        self.llm = self._create_llm(model)
        self.embeddings = self._create_embeddings()
        self.vectorstore = self._connect_vectorstore()
        self.retriever = self._create_retriever()
        self.prompt = self._create_prompt()
        self.rag_chain = self._create_chain()
        self.store = {}

    def _create_llm(self, model: str):
        return ChatOpenAI(
            model=model,
            api_key=self.api_key,
            temperature=1.0
        )

    def _create_embeddings(self):
        return OpenAIEmbeddings(
            openai_api_key=self.api_key,
            model="text-embedding-3-small"
        )

    def _connect_vectorstore(self):
        return Chroma(
            persist_directory=self.db_path,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )

    def _create_retriever(self):
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

    def _create_prompt(self):
        return ChatPromptTemplate.from_template("""
           You are a research assistant chatbot.

            Use conversation history and context to answer.
            If you not know the answer, say I am sorry do not have answer
                                                
            Conversation history:
            {chat_history}
        3 
            Context:
            {context}

            Question:
            {question}

            Answer:
            - Clear explanation
            - Include sources like ``Source: filename (Section, Page Number)```
        """)

    def get_session_history(self, session_id):
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def _create_chain(self):
        base_chain =  (
            {
                "context": itemgetter("question") | self.retriever | 
                (lambda docs: "\n\n".join(
                    f"Source: {doc.metadata.get('source', 'unknown')}\nPage Number: {doc.metadata.get('page_label', 'unknown')}\nContent: {doc.page_content}"
                    for doc in docs
                )),
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
            }   
            | self.prompt| self.llm | StrOutputParser()
        )

        return RunnableWithMessageHistory(
            base_chain,
            self.get_session_history,
            input_messages_key = "question",
            history_messages_key = "chat_history"
        )

    def ask(self, question: str, session_id: str="default") -> str:
        return self.rag_chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        )

    def chat(self):
        while True:
            question = input("How can I help you?\n").strip()
            if question.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            response = self.ask(question)
            print('\n--------------------------- ANSWER ---------------------------')
            print(response)
            print()
            print("=" * 40)

#
# if __name__ == "__main__":
#     chatbot = ChatBot()
#     chatbot.chat()