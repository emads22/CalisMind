import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from a .env file for securely managing sensitive keys and settings
load_dotenv()

# Set the OpenAI API key from environment variables
# Ensure the .env file contains a line like: OPENAI_API_KEY=your_openai_api_key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Define the OpenAI model to use
# gpt-4o: Optimized for better performance; gpt-4o-mini: Lower-cost alternative
OPENAI_MODEL = "gpt-4o"  # Alternatively for lower costs: "gpt-4o-mini"

# Define paths for storing the vector database and knowledge base
# Path to store the vector database (Chroma store)
DB_PATH = Path("./calismind_db")
# Root directory for the knowledge base (PDFs, etc.)
KNOWLEDGE_BASE_DIR = Path("./calisthenics_knowledge_base")

# Define chunking parameters for document processing
CHUNK_SIZE = 1000  # Number of characters per chunk
CHUNK_OVERLAP = 200  # Number of overlapping characters between consecutive chunks

"""
Recommended Chunking Values for Common Use Cases:

- Question Answering:
    . Chunk Size: 1000-1500
    . Overlap: 200-300

- Summarization:
    . Chunk Size: 1000-1500
    . Overlap: 200-300

- Classification:
    . Chunk Size: 500-1000
    . Overlap: 50-100

- Information Retrieval (RAG):
    . Chunk Size: 800-1200
    . Overlap: 150-250
"""

# Number of top chunks to retrieve from the vector store for each query.
K_RESULTS = 25
"""
Suggestions for K_RESULTS:
- Small Knowledge Base (e.g., < 500 chunks): Use a smaller value like 5-10 to reduce unnecessary retrieval.
- Large Knowledge Base (e.g., > 5000 chunks): Use a larger value like 25-50 for better recall.
- Question Answering: A value of 15-30 is recommended to capture enough context without overwhelming the LLM.
- Summarization: Lower values (5-10) are often sufficient since the focus is on summarizing key points.
- Experiment: Start with 25 as a baseline and adjust based on retrieval performance and LLM output quality.
"""


# Define CLI interface information
VECTORIZE_CLI_TITLE = "CLI for managing document processing, vector store creation, and statistics."

VECTORIZE_CLI_CHOICES = [
    "Load documents, add metadata, and split into chunks",  # Option 1
    "Print statistics about documents and chunks",          # Option 2
    "Create and persist a vector store",                    # Option 3
    "Load an existing vector store",                        # Option 4
    "Print statistics about the vector store",              # Option 5
    "Exit"                                                  # Option 6
]

# Notes for Developers:
# - Ensure the `calisthenics_knowledge_base` directory exists and contains the documents (PDFs, etc.) to process.
# - Adjust `DB_PATH` and `KNOWLEDGE_BASE_DIR` as needed to suit your directory structure.
# - Use CHUNK_SIZE and CHUNK_OVERLAP values based on the specific use case for optimal performance.
# - The CLI menu options are defined in VECTORIZE_CLI_CHOICES and correspond to actions in the interactive CLI.
