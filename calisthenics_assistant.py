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


def test_question(conversation_chain, question):
    """
    Asks a question to the conversational retrieval chain and prints the answer.

    Args:
        conversation_chain (ConversationalRetrievalChain): The initialized conversation chain.
        question (str): The question to ask, related to the calisthenics knowledge base.

    Returns:
        str: The answer retrieved from the conversation chain.
    """
    print("\n📝 Question:", question)

    # Invoke the conversation chain with the provided question
    result = conversation_chain.invoke({"question": question})
    answer = result["answer"]

    # Print the answer in a formatted way
    print("\n✅ Answer:", answer)
    return answer


if __name__ == "__main__":
    try:
        # Initialize the conversation chain
        conversation_chain = initialize_conversation_chain()

        # Example question related to calisthenics
        question = "What are the benefits of pull-ups for upper body strength?"
        answer = test_question(conversation_chain, question)

    except ValueError as e:
        print(e)
