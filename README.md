
# CalisMind: Unlock the Art of Bodyweight Training

![CalisMind_logo](./assets/images/CalisMind_logo.png)

## Overview

**CalisMind** is an intelligent, AI-powered application crafted to revolutionize the learning experience for calisthenics enthusiasts. By leveraging state-of-the-art language models, it transforms vast calisthenics knowledge bases into an interactive and accessible resource. Users can engage in real-time Q&A sessions, explore specific techniques, and retrieve detailed insights, all tailored to their unique fitness goals and queries.

Whether you're a beginner looking to master the basics or a seasoned athlete aiming to refine advanced techniques, CalisMind offers unparalleled support. It provides detailed explanations, the benefits of exercises, training programs, and expert advice to help users enhance their understanding and performance in calisthenics. Designed with flexibility and adaptability in mind, CalisMind empowers users to unlock the full potential of their fitness journey with ease and confidence.


---

- **Conversational AI**:
  Engage with an AI-powered chatbot trained on calisthenics resources for tailored advice and learning.
  
- **Retrieval-Augmented Generation (RAG)**:
  Combines knowledge retrieval with OpenAI's LLMs to provide accurate and context-aware answers.

- **Dynamic Knowledge Base**:
  Processes and organizes content from PDFs, making it accessible and searchable.

- **Interactive Gradio Interface**:
  A user-friendly web interface to chat with the AI and explore the calisthenics knowledge base.

- **Customizable Chunking**:
  Configurable document chunk sizes and overlap for optimized knowledge retrieval.

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
   
2. **Paths**:
   - `DB_PATH`: Path to store the vector database. Adjust if you want to use a different directory.
   - `KNOWLEDGE_BASE_DIR`: Path to your local knowledge base (e.g., PDFs). Ensure this directory exists and contains the documents you want to process.

3. **Document Chunking**:
   - `CHUNK_SIZE` and `CHUNK_OVERLAP`: Control how documents are split into smaller pieces for retrieval. Refer to the comments in `config.py` for recommended values based on your use case (e.g., question answering, summarization, or information retrieval).

4. **Retriever Settings**:
   - `K_RESULTS`: Defines the number of top chunks retrieved for each query. Adjust based on the size of your knowledge base and desired performance.

These settings are documented in `config.py` with detailed suggestions and recommendations to help you tailor the application to your needs.

### Model Customization
CalisMind is designed to be flexible and supports a wide range of AI models, whether open-source, closed-source, or even custom-built models. While the default implementation uses OpenAI's `gpt-4o` model for its conversational and retrieval-augmented generation features, users are free to choose any model that suits their needs. 

To integrate your preferred model:
1. **Closed-Source Models (e.g., OpenAI, Anthropic)**:
   - Obtain the required API keys from the respective provider.
   - Add the API key to your `.env` file. For example:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     ```

2. **Open-Source Models (e.g., Hugging Face Models)**:
   - Ensure you have a Hugging Face account and a valid access token.
   - Add the token to your `.env` file to authenticate and load models from the Hugging Face Hub:
     ```env
     HUGGINGFACEHUB_API_TOKEN=your_huggingface_access_token
     ```

3. **Custom Models**:
   - If you're using a locally hosted model or a custom model, ensure it can be accessed via an API endpoint or directly loaded in the codebase.
   - Modify the `initialize_conversation_chain` function or similar setup to use your custom model.

**Important**: The flexibility of CalisMind allows users to experiment with different models. Just ensure that the appropriate credentials or configurations are provided for seamless integration.

---

## Usage

### Running the App
1. Navigate to the project directory.
2. Launch the application:
   ```bash
   python app.py
   ```
3. The app will open in your default web browser. If it doesn’t, copy and paste the URL provided in the terminal into your browser.

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



