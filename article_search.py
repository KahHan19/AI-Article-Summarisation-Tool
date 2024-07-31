import os
import streamlit as st
import pickle
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate


def main():

    # Streamlit Buttons-Setups
    st.sidebar.title("Article Search")
    st.sidebar.write("Enter up to 3 links and a question to search through the articles.")

    question = st.sidebar.text_input("Question:")
    urls = []

    for i in range(3):
        url = st.sidebar.text_input(f"Article Link {i+1}:")
        urls.append(url)

    main_placefolder = st.empty()


    # Prompt_Setups
    prompt_template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {summaries}
    QUESTION: {question}
    SOURCES:
    FINAL ANSWER:
    """
    doc_prompt_template = """
    Content: {page_content}
    Source: {source}
    """

    DOC_PROMPT = PromptTemplate(
        template=doc_prompt_template, input_variables=["page_content", "source"])

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["summaries", "question"]
    )


    # Parameters
    callbacks=[StreamingStdOutCallbackHandler()]
    local_path = r"path\to\model"# Ensure that the model ends with ".gguf" so that it is compatible to run locally
    llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True) 






    if st.sidebar.button("Process Links"):

        # Loading Data from urls

        main_placefolder.text("Links being processed...")


        loader = UnstructuredURLLoader(urls= urls)
        data = loader.load()

        doc_split = RecursiveCharacterTextSplitter(
        separators= ["\n\n", "\n", ".", " "], # List of seperators
        chunk_size = 1000, # size of each chunk created
        chunk_overlap = 100, # size of  overlap between chunks in order to maintain the context
        length_function = len
        )

        docs = doc_split.split_documents(data)

        main_placefolder.text("Tokenizing the data...")

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        vectorindex_openai = FAISS.from_documents(docs, embeddings)

        
        # Pickle Save
        file_path = "vector_index.pkl"

        with open(file_path, "wb") as f:
            pickle.dump(vectorindex_openai, f)

        main_placefolder.text("Processing the question...")

        if question:
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    vectorstore = pickle.load(f)
                    chain_type_kwargs = {"prompt": PROMPT, "document_prompt": DOC_PROMPT }
                    chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff",retriever=vectorstore.as_retriever(),chain_type_kwargs=chain_type_kwargs,return_source_documents=True,verbose=True)
                    answer = chain({"question": question}, return_only_outputs=True)

                    main_placefolder.text(" ")
                    st.header("Answer")
                    st.write(answer["answer"])


                    st.header("Source")

                    for i in range(3):
                        if answer["source_documents"][i]:
                            st.write(f"Source[{i}]: " +answer["source_documents"][i].metadata["source"])
                        else:
                            st.write(f"Source[{i}]: None")

if __name__ == "__main__":
    main()




                    





















