import streamlit as st
from helper import get_pdf_text, get_text_chunks, get_vectorstore, get_conversational_retrieval_chain

def main():
    st.set_page_config(page_title="Information Retrieval System")
    st.header("📄 Information Retrieval System 💁‍♂️🙋")

    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar - PDF Upload Section
    with st.sidebar:
        st.title("📂 Menu")
        pdf_docs = st.file_uploader(
            "Upload PDF Files and Click On Submit & Process Button",
            type=["pdf"],
            accept_multiple_files=True
        )
        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing your documents..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversational_retrieval_chain(vectorstore)
                st.success("✅ Files uploaded and processed successfully!")

    # Chat Interface - Only visible after processing
    if st.session_state.conversation:
        st.subheader("💬 Chat with your Documents")

        user_question = st.text_input("Ask a question:")
        if user_question:
            with st.spinner("Generating response..."):
                response = st.session_state.conversation.invoke({"question": user_question})
                st.session_state.chat_history = response['chat_history']
                st.success("✅ Response generated!")

            # Display chat history
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.markdown(f"**👤 User:** {message.content}")
                else:
                    st.markdown(f"**🤖 Reply:** {message.content}")
    else:
        st.info("👈 Please upload and process PDF files from the sidebar to start chatting.")

if __name__ == "__main__":
    main()
