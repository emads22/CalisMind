import os
import openai
import gradio as gr
from dotenv import load_dotenv
from vectorize import load_vector_store
from rag_setup import user_prompt
from config import OPENAI_MODEL, SYSTEM_PROMPT, MAX_TOKENS, TEMPERATURE, UI_CSS


def chat(user_input, history):
    """
    Handles a chat interaction by building the conversation context, generating the assistant's 
    response in a streaming manner, and updating the chat history.

    Args:
        user_input (str): The user's input message or question.
        history (list): The chat history in "messages" format, where each message is a dictionary 
                        with "role" (e.g., "user", "assistant") and "content" keys.

    Yields:
        list: The updated chat history in "messages" format, including the streaming response.
    """
    # Initialize the messages list with the system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Add the existing history and the current user input with retrieval context to the messages list
    messages += history + \
        [{"role": "user", "content": user_prompt(user_input, retriever)}]

    # Create the chat completion stream using OpenAI's API
    stream = openai.chat.completions.create(
        model=OPENAI_MODEL,         # The model to use (e.g., "gpt-4")
        messages=messages,          # The full conversation context
        stream=True,                # Enable streaming for real-time responses
        max_tokens=MAX_TOKENS,      # Maximum tokens for the response
        temperature=TEMPERATURE     # Control randomness in the response
    )

    # Add the current user input to the history
    history += [{"role": "user", "content": user_input}]

    # Initialize an empty string to accumulate the assistant's response
    response = ""
    for chunk in stream:
        # Extract the content of the current chunk and append it to the response
        response += chunk.choices[0].delta.content or ""

        # Yield the updated chat history with the current partial response
        yield history + [{"role": "assistant", "content": response}]


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
            fn=chat,
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
            fn=chat,
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
        # Load the vector store and create a retriever at app startup
        print("\n🔄 Loading the vector store and initializing the retriever...\n")
        vector_store = load_vector_store()
        retriever = vector_store.as_retriever()
        print("✅ Vector store and retriever successfully initialized!\n")
    except Exception as error:
        raise Exception(
            f"\n❌ Error: Failed to load the vector store or initialize the retriever.\n"
            f"🚨 Reason: {str(error)}\n"
        ) from error

    # Step 3: Launch the CalisMind application
    print("\n🚀 Launching the CalisMind application...\n\n")
    launch_app()
