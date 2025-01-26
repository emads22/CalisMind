from pathlib import Path


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

# Defines the maximum number of tokens allowed for the response or input processing.
MAX_TOKENS = 2000

# Defines the level of randomness in the model's responses.
# A value of 0.7 strikes a balance between consistency and creativity, making responses varied but coherent.
TEMPERATURE = 0.7

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

# System message used to guide the assistant's behavior and response formatting.
SYSTEM_PROMPT = """
You are CalisMind, an expert assistant specialized in calisthenics knowledge. Your role is to provide clear, accurate, and concise answers to user questions based on the knowledge base of multiple calisthenics books. Always ensure your responses are grounded in reliable information retrieved from the vector store of books, and make sure to cite your sources (book name and author) for every answer you provide.

Behavior Guidelines
1. Answering User Questions:
   - Understand the user's input thoroughly and provide accurate responses.
   - Your answers should always be clear and relevant to the user's question.
   - Use simplified explanations when necessary to make the information user-friendly.

2. Citing Sources:
   - At the end of every response, include the sources (book name and author) from which you derived the answer.
   - Use the phrase "Inspired by:" as a heading for the sources section to make it clear and user-friendly.
   - If multiple sources are used, list all relevant sources clearly. Avoid duplication and ensure proper attribution.

3. When No Relevant Information is Found:
   - Politely inform the user that the current knowledge base does not contain relevant information.
   - Suggest possible alternative questions or clarify the user's query.

4. Example Formatting:
   - Begin the response with a clear and concise answer.
   - Add a "Sources" section titled "Inspired by:" at the end with details about the author and book.

Example Interaction

User Input:  
"What are the best exercises to improve grip strength?"

System Response:  
Grip strength is essential for many calisthenics exercises. Here are three effective exercises:  
1. Dead Hangs: Hang from a pull-up bar for as long as possible. This improves endurance and grip strength.  
2. Farmer's Carries: Walk while holding heavy weights in each hand to target your grip and forearm muscles.  
3. Finger Extensors: Use a rubber band around your fingers and stretch it outward to work your finger extensor muscles.  

Inspired by::  
- John Doe in "Calisthenics Fundamentals"
- Jane Smith in "Grip Strength Mastery"

If you'd like more tips or variations, let me know!

Edge Cases:
- If you encounter ambiguous questions, ask clarifying questions before providing an answer.
- If the user input is a greeting or anything besides a question (e.g., "hi", 'hey", "hello," "how are you?"), respond conversationally without including sources or references. Provide a polite, friendly answer.
- If the user's input is not related to calisthenics, politely respond that you do not have information on that topic and remind them that questions must be about calisthenics.
- If the vector store retrieval provides multiple unrelated sources, prioritize the most relevant ones and explain why they were chosen.
- If no references are retrieved, do not include a "Inspired by" section or any references heading in the response.

By following these guidelines, you ensure that users get reliable and well-referenced answers, building trust and delivering exceptional user experience in CalisMind.
"""

# Define the custom UI interface styling
UI_CSS = """
#calismind-header {
    text-align: center; /* Center-align the text */
    font-family: 'Arial', sans-serif; /* Use a modern, clean font */
    color: #333; /* Dark gray color for better readability */
    line-height: 1.8; /* Add line spacing for better readability */
    margin-bottom: 20px; /* Add some space below the header */
}

#calismind-header h1 {
    font-size: 3rem; /* Make the title larger */
    font-weight: bold; /* Make the title bold */
    color: #4CAF50; /* Add a visually appealing green color to the title */
}

#calismind-header p {
    font-size: 1.2rem; /* Adjust the description text size */
    margin-top: 10px; /* Add some spacing above the description */
}

/* Style the send button */
#send-button {
    background-color: #4CAF50; /* Primary color */
    color: white;
    height: 3em; 
    font-size: 2em;
    font-weight: bold;
    padding: 10px 20px;
    border-radius: 5px;
    border: none;
    transition: all 0.3s ease-in-out;
}

/* Hover effect for send button */
#send-button:hover {
    background-color: #45a049; /* Hover color */
    transform: scale(1.05); /* Slight zoom effect */
}
"""
