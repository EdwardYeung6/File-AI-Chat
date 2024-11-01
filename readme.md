# Multi-File Intelligent Q&A System

## Introduction
------------
This is an intelligent document Q&A system developed in Python that supports reading and answering questions for multiple file formats. Users can upload different types of documents (including PDF, Word, PPT, Excel, TXT, etc.) and ask questions about the document content using natural language. The system utilizes large language models to analyze document content and provide relevant answers. Please note that the system will only answer questions related to the uploaded document content.

## How It Works
------------

The system processes user questions in the following steps:

1. File Reading: The system supports reading multiple file formats, including:
   - PDF documents (`.pdf`)
   - Word documents (`.docx`)
   - PowerPoint presentations (`.pptx`) 
   - Excel spreadsheets (`.xlsx/.xls`)
   - Text files (`.txt`)

2. Text Extraction: Extract text content from different file formats.

3. Text Chunking: Split the extracted text into smaller chunks for efficient processing.

4. Vectorization: Use language models to convert text chunks into vector representations (embeddings).

5. Similarity Matching: When users ask questions, the system compares the question with text chunks to find semantically relevant content.

6. Answer Generation: Pass the filtered relevant text chunks to the language model to generate answers based on document content.

## Installation
----------------------------
Please follow these steps to install:

1. Clone the repository to local

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project directory and add your OpenAI API key and API base URL (if using other server or provider):

   ```
   OPENAI_API_KEY="your-api-key"
   BASE_API_PATH="your-api-base-url"
   ```

## Run the Project
----------------------------

1. Navigate to the project directory:
   ```
   cd your-project-directory
   ```

2. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

3. Open the displayed local URL in your browser (usually http://localhost:8501)

4. Upload documents and start asking questions


## Reference Tutorial
----------------------------
This project is built based on the following tutorial:

https://github.com/alejandro-ao/ask-multiple-pdfs
