import gradio as gr
from calisthenics_assistant import initialize_conversation_chain

# Global variable to store the conversation chain
# Ensure the conversation chain is initialized only once during the app's lifetime.
conversation_chain = None


def setup_gradio_chat():
    """
    Sets up and launches a Gradio chat interface integrated with the conversation chain.
    Ensures that the conversation chain is only initialized once per app session.

    Returns:
        None
    """
    global conversation_chain

    # Initialize the conversation chain only if it's not already set
    if conversation_chain is None:
        print("\n🔄 Initializing the conversation chain...")
        conversation_chain = initialize_conversation_chain()
        print("\n✅ Conversation chain successfully initialized!")
    else:
        print("\nℹ️ Conversation chain already initialized.")

    def chat(question, history):
        """
        Handles the chat interaction by invoking the conversation chain.

        Args:
            question (str): The user's question or message.
            history (list): Chat history, not used directly here but maintained by Gradio.

        Returns:
            str: The response from the conversation chain.
        """
        result = conversation_chain.invoke({"question": question})
        return result["answer"]

    # Set up and launch the Gradio interface
    print("\n🚀 Launching Gradio Chat Interface...")
    gr.ChatInterface(chat, type="messages").launch(inbrowser=True)


# Call the function to launch the Gradio chat interface
if __name__ == "__main__":
    setup_gradio_chat()
