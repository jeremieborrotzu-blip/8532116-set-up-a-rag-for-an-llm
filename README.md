# RAG Assistant with Mistral

This project implements a virtual assistant based on the Mistral model, using the Retrieval-Augmented Generation (RAG) technique to provide accurate and contextual answers from a custom knowledge base.

## Features

- 🔍 **Semantic search** with FAISS to find the relevant documents
- 🧠 **Query classification** to determine whether a RAG search is necessary
- 🤖 **Answer generation** with the Mistral models (Small or Large)
- 📊 **Feedback visualization** with charts and statistics
- ⚙️ **Customizable settings** (model, number of documents, minimum score)

## Prerequisites

- Python 3.9+
- Mistral API key (obtained at [console.mistral.ai](https://console.mistral.ai/))

## Installation

1. **Clone the repository**

```bash
git clone <repo-url>
cd <repo-name>
```

2. **Create a virtual environment**

```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. **Install the dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure the API key**

Create a `.env` file at the root of the project with the following content:

```
MISTRAL_API_KEY=your_mistral_api_key
```

## Project structure

```
.
├── MistralChat.py          # Main Streamlit application
├── indexer.py              # Script to index the documents
├── inputs/                 # Folder for the source documents
├── vector_db/              # Folder for the FAISS index and the chunks
├── database/               # SQLite database for the interactions
├── utils/                  # Utility modules
│   ├── config.py           # Application configuration
│   ├── database.py         # Database management
│   ├── query_classifier.py # Query classification
│   └── vector_store.py     # Vector index management
└── pages/                  # Additional Streamlit pages
    └── 1_Feedback_Viewer.py # Feedback visualization
```

## Usage

### 1. Add documents

Place your documents in the `inputs/` folder. The supported formats are:
- PDF
- TXT
- DOCX
- CSV
- JSON

You can organize your documents into subfolders for better organization.

### 2. Index the documents

Run the indexing script to process the documents and create the FAISS index:

```bash
python indexer.py
```

This script will:
1. Load the documents from the `inputs/` folder
2. Split the documents into chunks
3. Generate embeddings with Mistral
4. Create a FAISS index for the semantic search
5. Save the index and the chunks in the `vector_db/` folder

### 3. Launch the application

```bash
streamlit run MistralChat.py
```

The application will be available at http://localhost:8501 in your browser.

## Main features

### Query classification

The application automatically determines whether a question requires a RAG search or whether a direct answer from the Mistral model is enough. This helps optimize the performance and the relevance of the answers.

### Customizable settings

In the sidebar, you can adjust:
- The Mistral model (Small or Large)
- The number of documents to retrieve (1-20)
- The minimum similarity score (0-100%)

### Feedback and analysis

The application logs the interactions and the users' feedback. You can view the statistics in the "Feedback Viewer" page.

## Main modules

### `utils/vector_store.py`

Manages the FAISS vector index and the semantic search:
- Loading and splitting the documents
- Generating the embeddings with Mistral
- Creating and querying the FAISS index

### `utils/query_classifier.py`

Determines whether a query requires a RAG search:
- Keyword analysis
- Classification with the Mistral model
- Detection of specific vs general questions

### `utils/database.py`

Manages the SQLite database for the interactions:
- Logging the questions and answers
- Storing the users' feedback
- Retrieving the statistics

## Customization

You can customize the application by changing the settings in `utils/config.py`:
- Mistral models used
- Chunk size and overlap
- Default number of documents
- Town or organization name
