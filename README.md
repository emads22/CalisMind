# CalisMind: Unlock the Art of Bodyweight Training

![CalisMind_logo](./assets/images/CalisMind_logo.png)

## Overview

**CalisMind** is an intelligent, AI-powered application crafted to revolutionize the learning experience for calisthenics enthusiasts. By leveraging state-of-the-art language models, it transforms vast calisthenics knowledge bases into an interactive and accessible resource. Users can engage in real-time Q&A sessions, explore specific techniques, and retrieve detailed insights, all tailored to their unique fitness goals and queries.

Whether you're a beginner looking to master the basics or a seasoned athlete aiming to refine advanced techniques, CalisMind offers unparalleled support. It provides detailed explanations, the benefits of exercises, training programs, and expert advice to help users enhance their understanding and performance in calisthenics. Designed with flexibility and adaptability in mind, CalisMind empowers users to unlock the full potential of their fitness journey with ease and confidence.

---

## Features

- **Conversational AI**:  
  Engage with an AI-powered chatbot trained on calisthenics resources for tailored advice and learning.
  
- **Retrieval-Augmented Generation (RAG)**:  
  Combines knowledge retrieval with OpenAI's LLMs to provide accurate and context-aware answers. The RAG pipeline integrates a vector database to retrieve the most relevant chunks of information from your knowledge base, ensuring the AI generates responses grounded in factual, context-specific content.

- **Dynamic Knowledge Base**:  
  Processes and organizes content from PDFs, making it accessible and searchable. Users can extend or replace the knowledge base with their own documents for personalized use.

- **Interactive Gradio Interface**:  
  A user-friendly web interface to chat with the AI and explore the calisthenics knowledge base.

- **Customizable Chunking**:  
  Configurable document chunk sizes and overlap for optimized knowledge retrieval. Adjust these settings to improve performance for various tasks like summarization or question answering.

- **Flexible Embedding Models**:  
  Choose between OpenAI embeddings or HuggingFace open-source models for vector creation, providing options for cost-effective, offline, and customizable usage.

---

## Setup

### Clone the Repository
Clone the repository to your local system:
```bash
git clone https://github.com/emads22/CalisMind.git
cd CalisMind
```

### Install Dependencies
Ensure all dependencies are installed by setting up the provided Conda environment:
```bash
conda env create -f calismind_env.yml
conda activate calismind
```

### Set Up API Keys
To use OpenAI models, you'll need to configure your API keys:
1. Obtain API keys from the OpenAI platform.
2. Create a `.env` file in the project directory with the following structure:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```
3. (Optional) If using additional models or services like Anthropic or Pinecone, add their keys to the `.env` file as well:
   ```env
   ANTHROPIC_API_KEY=your_anthropic_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

### Configuring `config.py`
The `config.py` file contains key variables that control the behavior and setup of the CalisMind application. Users are encouraged to adjust these variables to suit their specific use case:

1. **Model Selection**:
   - `OPENAI_MODEL`: Choose the OpenAI model to use (e.g., `"gpt-4o"` for optimal performance or `"gpt-4o-mini"` for lower costs).

2. **Embedding Model Selection**:
   - `HF_EMBEDDINGS_MODEL`: Define the HuggingFace embeddings model for vector creation. By default, the model is set to `"sentence-transformers/all-MiniLM-L6-v2"`, which balances speed and accuracy. Other popular options include:
     - `"sentence-transformers/all-mpnet-base-v2"`: Higher accuracy but more resource-intensive.
     - `"sentence-transformers/paraphrase-MiniLM-L12-v2"`: Optimized for paraphrase detection.
   - To switch between OpenAI embeddings and HuggingFace embeddings, you can configure the embedding logic in the script as per your needs.

3. **Paths**:
   - `DB_PATH`: Path to store the vector database. Adjust if you want to use a different directory.
   - `KNOWLEDGE_BASE_DIR`: Path to your local knowledge base (e.g., PDFs). Ensure this directory exists and contains the documents you want to process.

4. **Document Chunking**:
   - `CHUNK_SIZE` and `CHUNK_OVERLAP`: Control how documents are split into smaller pieces for retrieval. Refer to the comments in `config.py` for recommended values based on your use case (e.g., question answering, summarization, or information retrieval).

5. **Retriever Settings**:
   - `K_RESULTS`: Defines the number of top chunks retrieved for each query. Adjust based on the size of your knowledge base and desired performance.

6. **Adjustable Constants**:
   - `MAX_TOKENS`: Sets the maximum token limit for responses. Increase or decrease based on the expected response length and API limits.
   - `TEMPERATURE`: Controls the randomness of responses. Use lower values (e.g., `0.2`) for deterministic outputs and higher values (e.g., `0.8`) for more creative responses.
   - `SYSTEM_PROMPT`: Defines the behavior and tone of the assistant. Modify this to customize the assistant's responses and personality.

These settings are documented in `config.py` with detailed suggestions and recommendations to help you tailor the application to your needs.

---

## Usage

### Two Application Options
CalisMind offers two modes of operation, allowing users to choose based on their needs:

#### 1. **Run `app.py` for Full Control**
   - This mode allows fine-grained control over the vector store and retrieval process.
   - Features custom logic for:
     - Fetching responses.
     - Streaming results to the user.
   - Ideal for users who want to experiment with or customize the retrieval and response logic.

   **To run `app.py`:**
   ```bash
   python app.py
   ```
   This will launch the Gradio interface for interacting with CalisMind.

#### 2. **Run `langchain_app.py` for Simplified LangChain Abstractions**
   - Uses LangChain's built-in abstractions for managing the retrieval-augmented generation (RAG) pipeline.
   - Easier to set up and extend using LangChain's modular components.
   - Ideal for users who prefer leveraging LangChain's standard patterns without customizing low-level logic.

   **To run `langchain_app.py`:**
   ```bash
   python langchain_app.py
   ```
   This will also launch a Gradio interface for interaction.

- ### **Comparison of Options:**
  | Feature                       | `app.py`                          | `langchain_app.py`               |
  |-------------------------------|-----------------------------------|----------------------------------|
  | Vector Store Control          | Full control                      | Abstracted via LangChain         |
  | Retrieval Customization       | Customizable                      | Limited to LangChain patterns    |
  | Streaming Responses           | Built-in                          | LangChain defaults               |
  | Ease of Setup                 | Requires manual configuration     | Simpler with LangChain presets   |

### Customizing the Knowledge Base
Users are free to use their own knowledge base documents with CalisMind. The project provides a CLI script (`vectorize.py`) that allows you to process and manage custom documents locally. With this script, you can:

1. **Load Custom Documents**:
   - Add your PDF files or other supported formats to a specified directory.

2. **Split Documents into Chunks**:
   - The script automatically splits your documents into smaller, manageable chunks for optimized retrieval.

3. **Create and Persist a Vector Store**:
   - Generate a vector store tailored to your custom knowledge base, which can be stored locally for future use.

The pre-built vector store for Calisthenics has been included for your convenience, but users are encouraged to load other vector stores or create their own as needed.

To use the CLI, simply run:
```bash
python vectorize.py
```

This will guide you through the process of building a custom vector store for your documents.

For advanced users, the vector store can be extended or replaced entirely based on specific needs, ensuring flexibility and adaptability for various domains beyond calisthenics.

---

## Contributing
Contributions are welcome! Here are some ways you can contribute to the project:
- Report bugs and issues.
- Suggest new features or improvements.
- Submit pull requests with bug fixes or enhancements.

---

## Author
- **Emad**  
  [<img src="https://img.shields.io/badge/GitHub-Profile-blue?logo=github" width="150">](https://github.com/emads22)

---

## License
This project is licensed under the MIT License, which grants permission for free use, modification, distribution, and sublicense of the code, provided that the copyright notice (attributed to [emads22](https://github.com/emads22)) and permission notice are included in all copies or substantial portions of the software. This license is permissive and allows users to utilize the code for both commercial and non-commercial purposes.

Please see the [LICENSE](LICENSE) file for more details.
