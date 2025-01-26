import os
import gradio as gr
from dotenv import load_dotenv
from rag_setup import initialize_conversation_chain
from config import UI_CSS


def chat_as_tuples(user_question, history):
    """
    Handles chat interactions for Gradio's Chatbot component in the default "tuples" format.

    This format represents the chat history as a list of tuples, where each tuple contains 
    the user's message and the assistant's response:
        [(user_message, ai_response), ...]

    Args:
        user_question (str): The user's input question or message.
        history (list): The chat history represented as a list of tuples 
                        (e.g., [(user_message, ai_response), ...]).

    Returns:
        list: The updated chat history, including the user's input and the assistant's response,
              formatted as a list of tuples.
    """
    # Generate the assistant's response by invoking the conversation chain
    result = conversation_chain.invoke({"question": user_question})
    answer = result["answer"]

    # Add the user's question and the assistant's response as a tuple to the history
    history.append((user_question, answer))

    # Return the updated history in the "tuples" format
    return history
    # Note: Gradio's gr.Chatbot component defaults to this "tuples" format unless
    # the "type" parameter is explicitly set to "messages".


def chat_as_messages(user_question, history):
    """
    Handles chat interactions for Gradio's Chatbot component in the "messages" format.

    This format represents the chat history as a list of dictionaries, where each dictionary 
    specifies the role ('system', 'user', 'assistant') and the content of the message:
        [
            {"role": "user", "content": "Your message here"},
            {"role": "assistant", "content": "AI's response here"},
            ...
        ]

    Args:
        user_question (str): The user's input question or message.
        history (list): The chat history represented as a list of dictionaries, 
                        each with a "role" and "content" key.

    Returns:
        list: The updated chat history, including the user's input and the assistant's response, 
              formatted as a list of dictionaries.
    """
    # Add the user's question as a dictionary with role "user" to the history
    history.append({"role": "user", "content": user_question})

    # Generate the assistant's response by invoking the conversation chain
    result = conversation_chain.invoke({"question": user_question})
    answer = result["answer"]

    # Add the assistant's response as a dictionary with role "assistant" to the history
    history.append({"role": "assistant", "content": answer})

    # Return the updated history in the "messages" format
    return history
    # Note: When using gr.Chatbot(type="messages") or gr.ChatInterface(type="messages"),
    # this "messages" format is required for proper functionality and compatibility.


# Build the Gradio interface with Blocks
def launch_app():
    with gr.Blocks(css=UI_CSS) as ui:
        # Header with description, centered
        with gr.Row():
            gr.Markdown(
                """
                # 🤸‍♂️ CalisMind
                Welcome to **CalisMind**, your AI-powered assistant for all things calisthenics!  
                Whether you're a beginner learning the basics or an advanced athlete refining techniques, 
                CalisMind is here to provide expert guidance.  
                Get detailed answers, explore training programs, and unlock the secrets to mastering calisthenics!  
                """,
                elem_id="calismind-header"
            )

        # Chat Interface with a Label
        with gr.Row():
            # Define the Chatbot component
            chatbot = gr.Chatbot(label="CalisMind Chat", type="messages")
            # By default, Gradio's gr.Chatbot uses "tuples" format for conversation history.
            # - If type="tuples" (default): Use the chat_as_tuples() function, which handles history as [(user_message, ai_response), ...].
            # - If type="messages": Use the chat_as_messages() function, which handles history as [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}, ...].
            # - Choose type="messages" if you need compatibility with APIs like OpenAI's Chat API, which expects the "messages" format.

        # Input box and Send button
        with gr.Row():
            with gr.Column(scale=4):  # Scale the Textbox to 3 parts
                user_input = gr.Textbox(
                    placeholder="What’s on Your Mind?...",
                    label="Ask CalisMind",
                    lines=1,
                )
            with gr.Column(scale=1):  # Scale the Button to 1 part
                send_button = gr.Button("Send", elem_id="send-button")

        # Link both the Enter key and Send button to the chat function
        user_input.submit(
            fn=chat_as_messages,
            inputs=[user_input, chatbot],
            outputs=[chatbot],
            show_progress=True,
        ).then(
            # Clear the input box after sending
            fn=lambda: "",
            inputs=None,
            outputs=[user_input]
        )

        send_button.click(
            fn=chat_as_messages,
            inputs=[user_input, chatbot],
            outputs=[chatbot],
            show_progress=True,
        ).then(
            # Clear the input box after sending
            fn=lambda: "",
            inputs=None,
            outputs=[user_input]
        )

    # Launch the interface
    ui.launch(inbrowser=True)


# Run the app
if __name__ == "__main__":
    """
    Entry point for running the script directly.
    This block handles:
    1. Loading environment variables securely from a .env file.
    2. Validating and setting the OpenAI API key required for OpenAI services.
    3. Launching the CalisMind Gradio application.
    """

    # Step 1: Load environment variables from the .env file
    print("\n\n🔄 Loading environment variables...\n")
    load_dotenv()

    # Step 2: Validate and set the OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError(
            "❌ [Error]: OpenAI API key is missing. Please add it to your .env file as OPENAI_API_KEY.\n"
        )
    os.environ["OPENAI_API_KEY"] = openai_api_key
    print("✅ OpenAI API key successfully loaded.\n")

    try:
        # Initialize the conversation chain
        print("\n🔄 Initializing the conversation chain...\n")
        conversation_chain = initialize_conversation_chain()
        print("\n✅ Conversation chain successfully initialized!\n")
    except Exception as error:
        raise Exception(
            f"\n❌ Error: Failed to initialize the conversation chain.\n"
            f"🚨 Reason: {str(error)}\n"
        ) from error

    # Step 3: Launch the CalisMind application
    print("\n🚀 Launching the CalisMind application...\n\n")
    launch_app()
