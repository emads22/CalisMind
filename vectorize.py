import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from config import (KNOWLEDGE_BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
                    DB_PATH, VECTORIZE_CLI_TITLE, VECTORIZE_CLI_CHOICES,
                    HF_EMBEDDINGS_MODEL)


def add_metadata(doc, author_name, book_name):
    """
    Adds metadata (author and book name) to a document.

    Args:
        doc: A single document object to which metadata will be added.
        author_name: The name of the author extracted from the folder name.
        book_name: The name of the book extracted from the file name.

    Returns:
        The document object with updated metadata.
    """
    doc.metadata["author"] = author_name
    doc.metadata["book"] = book_name

    return doc


def load_and_process_documents():
    """
    Loads documents from the knowledge base directory, adds metadata for the author and book name,
    and splits them into chunks for processing.

    Returns:
        Tuple:
            - documents: A list of document objects with metadata.
            - chunks: A list of document chunks created by splitting the original documents.
    """
    documents = []

    # Iterate through author folders in the knowledge base directory
    # Each folder is assumed to be named after an author
    for folder_path in KNOWLEDGE_BASE_DIR.glob("*"):
        # Extract author name from the folder name
        author_name = folder_path.stem.title()

        # Load all PDF files from the author's folder
        loader = DirectoryLoader(
            path=folder_path,
            glob="**/*.pdf",  # Match all PDF files in the folder
            loader_cls=PyPDFLoader
        )
        folder_docs = loader.load()  # Load documents using the specified loader

        # Add metadata (author and book name) to each loaded document
        for doc in folder_docs:
            # Extract the book name from the file name (without extension)
            book_name = Path(doc.metadata["source"]).stem.title()
            documents.append(add_metadata(doc, author_name, book_name))

    # Initialize a text splitter for creating chunks from documents
    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,  # Define chunk size for splitting
        chunk_overlap=CHUNK_OVERLAP  # Define overlap size for context preservation
    )

    # Split the documents into smaller chunks
    chunks = text_splitter.split_documents(documents)

    return documents, chunks


def create_vector_store(chunks):
    """
    Creates a vector store from the document chunks, embeds them using OpenAI embeddings,
    and persists the store to disk. If an existing vector store is found, it deletes it.

    Args:
        chunks: A list of document chunks to be embedded and stored.

    Returns:
        vector_store: The created vector store object.
    """
    # Attempt to load an existing vector store
    vector_store = load_vector_store()

    # If a vector store exists, delete its contents
    if vector_store:
        vector_store.delete_collection()

    # Create a new vector store by embedding the document chunks
    vector_store = Chroma.from_documents(
        documents=chunks,  # Provide the chunks for embedding
        embedding=OpenAIEmbeddings(),  # Use OpenAI embeddings for vector creation
        persist_directory=str(DB_PATH)  # Directory to persist the vector store
    )

    return vector_store


def load_vector_store():
    """
    Loads an existing vector store from disk if it exists.

    Returns:
        vector_store: The loaded vector store object if it exists, or None otherwise.
    """
    # Check if the vector store directory exists
    if DB_PATH.exists():
        # Load the existing vector store
        vector_store = Chroma(
            # Directory where the vector store is persisted
            persist_directory=str(DB_PATH),
            embedding_function=OpenAIEmbeddings()  # Embedding function for consistency
        )
        return vector_store
    else:
        # Return None if no vector store exists
        return None


def create_vector_store_hf(chunks):
    """
    Creates a vector store from document chunks using Hugging Face embeddings.

    This function generates and persists a vector store using Hugging Face embeddings,
    offering a local, cost-free alternative to OpenAI embeddings. It checks for an
    existing vector store and clears its contents if found before creating a new one.
    Ideal for offline or budget-conscious environments.

    Args:
        chunks (list): A list of document chunks to be embedded and stored.

    Returns:
        vector_store (Chroma): The created vector store object.
    """
    db_path = Path(
        f"{str(DB_PATH)}_hf")  # Define the path for the Hugging Face vector store

    # Step 1: Initialize Hugging Face embeddings using the specified model
    hf_embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDINGS_MODEL)

    # Step 2: Attempt to load an existing vector store
    vector_store = load_vector_store_hf()

    # Step 3: If an existing vector store is found, delete its contents
    if vector_store:
        vector_store.delete_collection()

    # Step 4: Create a new vector store using the Hugging Face embeddings
    vector_store = Chroma.from_documents(
        documents=chunks,              # Preprocessed chunks to be embedded
        embedding=hf_embeddings,       # Hugging Face embeddings function
        # Directory for storing the vector store
        persist_directory=str(db_path)
    )

    return vector_store
    # Notes:
    # - Install `sentence-transformers` if not already installed: pip install sentence-transformers
    # - Popular HuggingFace models:
    #     * "sentence-transformers/all-MiniLM-L6-v2" (lightweight and fast)
    #     * "sentence-transformers/all-mpnet-base-v2" (higher accuracy, more resource-intensive)
    # - This function is a direct alternative to OpenAI embeddings for cost-effective and offline usage.


def load_vector_store_hf():
    """
    Loads an existing vector store that was created using Hugging Face embeddings.

    This function attempts to locate and load a vector store from the predefined
    directory. If the directory exists, the function initializes the store with 
    Hugging Face embeddings. If no vector store exists, it returns `None`.

    Returns:
        vector_store (Chroma or None): The loaded vector store object, or `None`
                                       if no existing vector store is found.
    """
    db_path = Path(
        f"{str(DB_PATH)}_hf")  # Define the path for the Hugging Face vector store

    # Step 1: Initialize Hugging Face embeddings using the specified model
    hf_embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDINGS_MODEL)

    # Step 2: Check if the vector store directory exists
    if db_path.exists():
        # Load the vector store if the directory is found
        vector_store = Chroma(
            persist_directory=str(db_path),
            embedding_function=hf_embeddings
        )
        return vector_store
    else:
        # Return None if no vector store exists
        return None


def document_stats(documents, chunks):
    """
    Prints statistics about the loaded documents and their chunks.

    Args:
        documents (list): A list of documents loaded from the knowledge base, 
                          each with metadata such as author and book name.
        chunks (list): A list of chunks generated by splitting the documents.

    Returns:
        None: Prints the document and chunk statistics directly.
    """
    # Check if documents or chunks are empty or None
    if not documents:
        print(
            "\n[Error] No documents found. Please ensure documents are loaded before fetching stats.\n")
        return
    if not chunks:
        print("\n[Error] No chunks found. Please ensure documents have been split into chunks before fetching stats.\n")
        return

    # Total number of documents and chunks
    print(f"\n{'=' * 40}")
    print(f"Document and Chunk Summary")
    print(f"{'=' * 40}")
    print(f"- Total number of original documents: {len(documents)}")
    print(f"- Total number of chunks (after splitting): {len(chunks)}")

    # Unique authors
    authors = set(doc.metadata["author"] for doc in documents)
    print(f"- Unique authors found:")
    for idx, author in enumerate(authors, start=1):
        print(f"\t{idx}. {author}")

    # Unique books
    books = set(doc.metadata["book"] for doc in documents)
    print(f"- Unique books found:")
    for idx, book in enumerate(books, start=1):
        print(f"\t{idx}. {book}")
    print(f"{'=' * 40}\n")


def vector_store_stats(vector_store):
    """
    Prints statistics about the vector store, including the number of vectors and their dimensions.

    Args:
        vector_store (Chroma): The vector store object containing processed document embeddings.

    Returns:
        None: Prints the vector store statistics directly.
    """
    # Check if the vector store has an internal collection
    if not vector_store or not hasattr(vector_store, "_collection") or not vector_store._collection:
        print("\n[Error] Vector store does not exist or is empty. Please ensure it is created before fetching stats.\n")
        return

    # Access the internal collection of vectors
    collection = vector_store._collection
    count = collection.count()  # Total number of vectors in the store

    # Retrieve a sample embedding to determine the dimensionality
    sample_embedding = collection.get(limit=1, include=["embeddings"])[
        "embeddings"][0]
    dimensions = len(sample_embedding)

    # Print vector store statistics
    print(f"\n{'=' * 40}")
    print(f"Vector Store Summary")
    print(f"{'=' * 40}")
    print(f"- Number of chunks (processed documents): {count:,}")
    print(f"- Number of vectors in the store: {count:,}")
    print(f"- Vector dimensionality: {dimensions:,}")
    print(f"{'=' * 40}\n")


def main():
    """
    Interactive CLI to let the user choose an action and call the corresponding function.
    """

    # Initialize global variables
    documents, chunks, vector_store = None, None, None

    while True:
        # Display the menu
        print(f"\n\n{'=' * 80}")
        print(f"   {VECTORIZE_CLI_TITLE}")
        print(f"{'=' * 80}\n")

        for idx, option in enumerate(VECTORIZE_CLI_CHOICES, start=1):
            print(f"  {idx}. {option}")

        # Get the user's choice
        choice = input("\n\n=> Enter your choice (1-8): ")

        # Handle the user's choice
        if choice == "1":
            print(
                "\n\n🔄 Loading documents, adding metadata, and splitting into chunks...")
            documents, chunks = load_and_process_documents()
            print(
                "\n✅ [Success]: Documents and chunks successfully loaded and processed!")

        elif choice == "2":
            print("\n\n📊 Printing statistics about documents and chunks...")
            if documents is None or chunks is None:
                print("\n❌ [Error]: Please load the documents first (Option 1).")
            else:
                document_stats(documents, chunks)
                print("\n✅ [Success]: Document and chunk statistics displayed!")

        elif choice == "3":
            print("\n\n🛠️ Creating a vector store from chunks...")
            if chunks is None:
                print(
                    "\n❌ [Error]: Please load and split documents first (Option 1).")
            else:
                # Create a vector store from the chunks (or documents if preferred)
                # The vector store embeds the chunks/documents and persists them for later retrieval
                # Alternatively: create_vector_store(documents)
                vector_store = create_vector_store(chunks)
                print(
                    "\n✅ [Success]: Vector store successfully created and persisted!")

        elif choice == "4":
            print("\n\n📂 Loading an existing vector store...")
            vector_store = load_vector_store()
            if vector_store:
                print("\n✅ [Success]: Vector store successfully loaded!")
            else:
                print(
                    "\n❌ [Error]: No vector store found. Please create one first (Option 3).")

        elif choice == "5":
            print(
                "\n\n🛠️ Creating a vector store using a Hugging Face open-source model..."
            )
            if chunks is None:
                print(
                    "\n❌ [Error]: No document chunks found. Please load and split documents first (Option 1)."
                )
            else:
                vector_store = create_vector_store_hf(chunks)
                print(
                    "\n✅ [Success]: Vector store successfully created using a Hugging Face model and persisted!"
                )

        elif choice == "6":
            print(
                "\n\n📂 Loading an existing vector store created with a Hugging Face open-source model..."
            )
            vector_store = load_vector_store_hf()
            if vector_store:
                print(
                    "\n✅ [Success]: Vector store successfully loaded using a Hugging Face model!")
            else:
                print(
                    "\n❌ [Error]: No existing vector store found. Please create one first using a Hugging Face model (Option 5)."
                )

        elif choice == "7":
            print("\n\n📈 Printing vector store statistics...")
            if vector_store is None:
                print(
                    "\n❌ [Error]: Please create or load a vector store first.")
            else:
                vector_store_stats(vector_store)
                print("\n✅ [Success]: Vector store statistics displayed!")

        elif choice == "8":
            print("\n\n👋 Exiting the CLI. Goodbye!")
            break

        else:
            print(
                "\n❌ [Error]: Invalid choice. Please enter a number between 1 and 8.")

    print(f"\n\n{'=' * 80}\n")


if __name__ == "__main__":
    """
    Entry point for running this script directly.
    Handles the following:
    1. Loads environment variables from a .env file for secure management of sensitive keys.
    2. Validates the presence of the OpenAI API key in the environment.
    3. Starts the main interactive CLI for document processing and vector store management.
    """

    # Step 1: Load environment variables from .env file
    print("\n🔄 Loading environment variables...")
    load_dotenv()

    # Step 2: Validate the OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError(
            "❌ [Error]: OpenAI API key is missing. Please add it to your .env file as OPENAI_API_KEY.\n"
        )
    os.environ["OPENAI_API_KEY"] = openai_api_key
    print("✅ OpenAI API key successfully loaded.")

    # Step 3: Start the main CLI
    print("\n🚀 Starting the CLI for document processing...\n")
    main()
