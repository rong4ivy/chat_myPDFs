
import os
import tempfile
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import PyPDF2

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "  "

class SimpleDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

def process_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def main():
    st.title("Chat with Multiple PDFs__Rong")

    uploaded_files = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    if uploaded_files and len(uploaded_files) > 0:
        all_texts = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name

            text = process_pdf(file_path)
            all_texts.append(text)

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        all_docs = []
        for text in all_texts:
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                all_docs.append(SimpleDocument(page_content=chunk))

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever())

        question = st.text_input("Ask something about the PDFs:")
        if st.button("Submit Query") and question:
            response = qa.run(question)
            st.session_state.conversation_history.append((question, response))

    if st.session_state.conversation_history:
        st.write("### Conversation History")
        for i, (q, a) in enumerate(st.session_state.conversation_history):
            st.write(f"**Q{i+1}:** {q}")
            st.write(f"**A{i+1}:** {a}")

if __name__ == "__main__":
    main()
