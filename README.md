# Conversational ChatBot For PDF

This project is a Streamlit-based application that allows users to upload PDF files and interact with their content using a conversational chatbot. The chatbot leverages various language models to answer questions and summarize the content of the PDFs.

## Features

- Upload multiple PDF files
- Choose from different language models
- Ask questions about the content of the PDFs
- Get concise answers and summaries
- View chat history

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd Chatwithpdf
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv myenv
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    Create a `.env` file in the root directory and add the following variables:
    ```env
    GROQ_API_KEY=<your-groq-api-key>
    LANGCHAIN_API_KEY=<your-langchain-api-key>
    LANGCHAIN_PROJECT=<your-langchain-project>
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run testCodeV3.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload PDF files, choose a language model, and start interacting with the content.

## License

This project is licensed under the MIT License.
