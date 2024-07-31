# AI Article Summarization Tool

Effortlessly extract and summarize information from news articles with our LLM tool. Simply provide article URLs and a relevant question to receive accurate, concise answers based on the content of the articles.

![UI](example.png)

## 🚀 Libraries and Technologies

- **GPT-4**: Provides advanced language understanding and generation.
- **LangChain**: Facilitates seamless integration and processing of language models.
- **Facebook AI Similarity Search (FAISS)**: Delivers efficient vector indexing and search for enhanced performance.
- **Streamlit**: Powers a user-friendly, interactive interface for a smooth experience.

## 🌟 Features

### Interactive Functionality
- **Search**: Input up to three article URLs for analysis.
- **Question**: Ask specific questions related to the content of these articles.
- **Answer**: Receive a summarized response with answers ranked by relevance.

### ⚙ Technical Requirements
- **Locally Hosted**: Ensures enhanced security by operating on local infrastructure.
- **User-Friendly Interface**: Built with Streamlit for an intuitive user experience.
- **GPT-4 and LangChain Integration**: Utilizes cutting-edge language models for improved query processing and accuracy.
- **FAISS Integration**: Provides efficient and fast vector indexing with Facebook AI Similarity Search.

## Model Details

- **Recursive Character Text Splitter**:
    1. **Improved Clarity**: Breaks large text into smaller chunks, isolating key information for easier retrieval.
    2. **Enhanced Efficiency**: Processes smaller chunks faster and in parallel, speeding up analysis.
    3. **Model Compatibility**: Ensures text chunks fit within the input size limits of machine learning models for effective processing.

- **Embedding Model (all-MiniLM-L6-v2)**:
    1. **Compact and Efficient**: Designed to be small and efficient, accommodating hardware constraints.

- **.gguf LLM Models**:
    1. **Local Execution**: .gguf files enable models to run on local CPUs.
    2. **Pre-trained Models**: Refer to Hugging Face for other pre-trained LLM models that may fit your needs.

## Future Work

- **Enhanced Models**: Explore more advanced pre-trained LLM models with additional layers and capabilities to handle more complex data processing.

