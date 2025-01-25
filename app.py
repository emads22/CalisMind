import gradio as gr
from calisthenics_assistant import initialize_conversation_chain
from config import UI_CSS

# Initialize the conversation chain
print("\n🔄 Initializing the conversation chain...\n")
conversation_chain = initialize_conversation_chain()
print("\n✅ Conversation chain successfully initialized!\n")


def chat(question, history):
    """
    Handles the chat interaction by invoking the conversation chain.
    Args:
        question (str): The user's question or message.
        history (list): The chat history maintained by Gradio.

    Returns:
        tuple: Updated chat history with the user's question and the AI's response.
    """
    result = conversation_chain.invoke({"question": question})
    answer = result["answer"]
    history.append((question, answer))  # Append the conversation to history
    return history, history  # Return the updated history for display

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
            chatbot = gr.Chatbot(label="CalisMind Chat")

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
            outputs=[chatbot, chatbot],
            show_progress=True,
        ).then(
            # Clear the input box after sending
            fn=lambda: "",
            inputs=None,
            outputs=[user_input]
        )

        send_button.click(
            chat,
            inputs=[user_input, chatbot],
            outputs=[chatbot, chatbot],
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
    launch_app()
