# PDF Chat

This is a Streamlit-based web application that allows users to upload PDF files and ask questions about their content. The application uses a combination of natural language processing techniques and vector-based text retrieval to provide answers to the user's questions.

## Features

1. **PDF Upload**: Users can upload one or more PDF files to the application.
2. **PDF Processing**: The application processes the uploaded PDF files, extracts the text content, and splits it into manageable chunks.
3. **Vector Store Creation**: The application creates a vector store using the OpenAI Embeddings library, which allows for efficient text retrieval.
4. **Question Answering**: Users can ask questions about the content of the uploaded PDF files, and the application will provide answers using the information in the vector store.
5. **Source Tracking**: The application not only provides the answer to the user's question but also indicates the sources (PDF pages) that were used to generate the answer.

## Technologies and Libraries Used

- **Streamlit**: A Python library for building interactive web applications.
- **PyPDF2**: A pure-python library built as a PDF toolkit.
- **langchain**: A toolkit for building applications with large language models.
- **OpenAI Embeddings**: A language model used for generating text embeddings.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors.
- **ChatOpenAI**: A language model from OpenAI used for generating responses to user questions.

## Installation and Usage

1. Clone the repository:
   ```
   git clone https://github.com/your-username/pdf-chat.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set the OpenAI API key as a Streamlit secret:
   ```
   streamlit secrets set OPENAI_API_KEY=your_openai_api_key
   ```

4. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

5. The application will open in your default web browser. You can then upload PDF files and ask questions about their content.

## Customization

You can customize the application by modifying the Python code in the `app.py` file. For example, you can change the language model, the text splitting algorithm, or the prompts used for generating the responses.

By email: rayyanahmed265@yahoo.com