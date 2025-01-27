from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from vectorize import load_vector_store
from config import OPENAI_MODEL, K_RESULTS


def initialize_conversation_chain():
    """
    Initializes a conversational retrieval chain with OpenAI's chat model, 
    a conversation buffer memory, and a vector store retriever for 
    Retrieval-Augmented Generation (RAG).

    This setup enables an LLM-powered conversational agent that retrieves relevant
    information from a vector store while maintaining conversation context.

    Returns:
        ConversationalRetrievalChain: The initialized conversation chain.
    """
    # Step 1: Create a ChatOpenAI instance
    # The temperature controls the randomness of responses (lower = more deterministic)
    # Model name is fetched from the configuration (e.g., "gpt-4o")
    llm = ChatOpenAI(
        temperature=0.7,
        model_name=OPENAI_MODEL
    )

    # Step 2: Set up a conversation buffer memory
    # This memory tracks the conversation history and allows the model to generate responses
    # with context-awareness.
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # Memory key used in the conversation chain
        return_messages=True       # Ensures the memory returns the full conversation history
    )

    # Step 3: Load the vector store and retrieve its retriever
    # The vector store is a pre-created database of document embeddings.
    vector_store = load_vector_store()
    if not vector_store:
        raise ValueError(
            "[Error] Vector store could not be loaded. Ensure it is created first.")
    # Convert the vector store into a retriever
    # The retriever abstracts the process of searching the vector store for
    # the top-k most relevant chunks based on a query.
    # "k" specifies the number of results to retrieve for each query.
    retriever = vector_store.as_retriever(search_kwargs={"k": K_RESULTS})

    # Step 4: Create a Conversational Retrieval Chain
    # This combines the language model (LLM), retriever, and memory into a single pipeline
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    return conversation_chain


def test_conversation_chain(question):
    """
    Tests the conversational retrieval chain by asking a question and printing the response.

    Args:
        question (str): The question to ask, related to the calisthenics knowledge base.

    Returns:
        None: This function does not return a value. The answer is printed to the console.

    Raises:
        Exception: If there is an issue initializing the conversation chain or processing the question,
                   the error message is caught and displayed.
    """
    try:
        # Step 1: Initialize the conversational retrieval chain
        # This sets up the chain responsible for retrieving and answering questions
        print("\n🔄 Initializing the conversation chain...")
        conversation_chain = initialize_conversation_chain()
        print("\n✅ Conversation chain successfully initialized!")

        # Step 2: Print the question for context
        print("\n📝 Question:", question)

        # Step 3: Invoke the conversation chain with the provided question
        # The chain processes the question and retrieves the answer
        result = conversation_chain.invoke({"question": question})
        answer = result["answer"]

        # Step 4: Print the retrieved answer in a formatted way
        print("\n✅ Answer:", answer)

    except Exception as e:
        # Handle and display any errors that occur during execution
        print(f"\n[Error] - {str(e)}\n")


def user_prompt(user_input, retriever):
    """
    Generates a formatted prompt based on the user's input question, 
    incorporating relevant references retrieved from the vector store.

    Args:
        user_input (str): The user's input question or query.
        retriever: The retriever object used to query the vector store 
                   and retrieve relevant documents.

    Returns:
        str: A formatted string that includes:
             - The user's input as "User Input".
             - A "Sources" section listing relevant references (author and book name).
               Each reference is formatted as:
               '- {author} in "{book}"'
             - If no references are retrieved, only the user's input is returned 
               without a "Sources" section.
    """
    # Retrieve results from the global retriever using the user input
    results = retriever.invoke(user_input)

    # Handle the case where no results are retrieved from the vector store
    if not results:
        return f"User Input: {user_input}"

    # Initialize a list to store formatted references
    retrieved_references = []
    for doc in results:
        # Extract metadata from each document, with fallbacks for missing metadata
        author = doc.metadata.get("author", "Unknown Author")
        book = doc.metadata.get("book", "Unknown Book")
        # Append the formatted reference to the list
        retrieved_references.append(f'- {author} in "{book}"')

    # Remove duplicate references and format the list as a string
    formatted_references = "\n".join(set(retrieved_references))

    # Build the final prompt, including the user's input and the sources
    return f"User Input: {user_input}\n\nSources:\n{formatted_references}."


def test_retriever(question):
    """
    Tests a conversational chain by initializing a vector store, 
    creating a retriever, and generating a prompt based on the user's question.

    Args:
        question (str): The user's input question to be used in the conversational chain.

    Steps:
        1. Load the vector store and initialize a retriever.
        2. Print the input question for context.
        3. Generate a user prompt using the question and the retriever.
        4. Print the generated prompt to verify the retrieval process.
    """
    # Step 1: Print a message indicating that the vector store is being loaded and the retriever is being initialized
    print("\n🔄 Loading the vector store and initializing the retriever...\n")

    # Load the vector store (assumed to be a prebuilt storage of embedded documents)
    vector_store = load_vector_store()

    # Convert the vector store into a retriever object to perform similarity-based queries
    retriever = vector_store.as_retriever()

    # Print a confirmation message once the vector store and retriever are successfully initialized
    print("✅ Vector store and retriever successfully initialized!\n")

    # Step 2: Print the question provided by the user for reference
    print("\n📝 Question:", question)

    # Step 3: Generate the user prompt by processing the input question based on the question and retrieval context
    prompt = user_prompt(question, retriever)

    # Step 4: Print the final formatted user prompt that will be passed to the AI model
    print("\n✅ User Prompt:", prompt)


if __name__ == "__main__":
    # Define an example question related to calisthenics for testing
    question = "What are the benefits of pull-ups for upper body strength?"

    # Test the conversational retrieval chain with the example question
    # test_conversation_chain(question)

    # Test the retriever with the example question
    test_retriever(question)
