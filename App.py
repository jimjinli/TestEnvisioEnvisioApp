import boto3
from typing import List
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

class ChatMessage():
    def __init__(self, role: str, text: str):
        self.role = role
        self.text = text

def login_screen():
    st.header("This app is private.")
    st.subheader("Please log in.")
    st.button("Log in with Google", on_click=st.login)

@st.cache_resource
def get_boto3_agent_runtime_client():
    session = boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )

    return session.client("bedrock-agent-runtime", region_name=os.getenv('AWS_DEFAULT_REGION'))

def chat_with_model(user_name: str, message_history: List[ChatMessage], new_text: str) -> None:
    bedrock_agent_runtime_client = get_boto3_agent_runtime_client()

    model_arn = os.getenv('MODEL_ARN')
    knowledge_base_id = os.getenv('KNOWLEDGE_BASE_ID')

    new_text_message = ChatMessage('user', text=new_text)
    message_history.append(new_text_message)

    response = bedrock_agent_runtime_client.retrieve_and_generate(
        input={
            'text': f'I am {user_name}. {new_text}'
        },
        retrieveAndGenerateConfiguration={
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': knowledge_base_id,
                'modelArn': model_arn
            }
        },
    )

    generated_text = response['output']['text']
    response_chat_message = ChatMessage('assistant', generated_text)

    message_history.append(response_chat_message)

def main() -> None:
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Envisio Envisio Knowledge Base")
    st.title("Envisio Envisio Knowledge Base")

    if not st.experimental_user.is_logged_in:
        login_screen()
    else:
        st.header(f"Welcome, {st.experimental_user.name}!")
        st.button("Log out", on_click=st.logout)

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        chat_container = st.container()

        input_text = st.chat_input("How may I help you?")

        if input_text:
            chat_with_model(user_name=st.experimental_user.name, message_history=st.session_state.chat_history, new_text=input_text)

        for message in st.session_state.chat_history:
            with chat_container.chat_message(message.role):
                st.markdown(message.text)

if __name__ == "__main__":
    main()
