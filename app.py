import os
import gradio as gr
from typing import List, Dict
import asyncio
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from base import Message
from core.chatbot import HustChatbot
from router.router import LangchainRouter
from rag.retrieval import SupabaseRetriever
from rag.responder import RAGResponder
from reflection.reflection import LangchainReflection
from rag.responder import ChitchatResponder
from langchain.chat_models import ChatOpenAI
from datetime import datetime
# from router import QuestionClassifier  # Giả sử bạn đã có class này

# Load environment variables
load_dotenv()

# Initialize components
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# router = QuestionClassifier(llm=llm)
router = LangchainRouter(llm=llm)
retriever = SupabaseRetriever()
reflection = LangchainReflection(llm=llm)
chitchat_responder = ChitchatResponder(llm=llm)
rag_responder = RAGResponder(llm=llm)

# Initialize chatbot
chatbot = HustChatbot(
    router=router,
    retriever=retriever,
    reflection=reflection,
    chitchat_responder=chitchat_responder,
    rag_responder=rag_responder
    )

async def process_response(message: str, history: List[Dict]) -> tuple:
    """Process message and get response"""
    try:
        # Process message
        response = await chatbot.process_message(message)
        # Wrap code blocks with ```
        if "```" not in response and any(lang in response.lower() for lang in ["python", "javascript", "java", "c++", "html", "css"]):
            response = f"```\n{response}\n```"
        history.append({"role": "assistant", "content": response})
        return "", history
    except Exception as e:
        error_msg = f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
        history.append({"role": "assistant", "content": error_msg})
        return "", history

def user_message(message: str, history: List[Dict]) -> tuple:
    """Add user message to history and process bot response"""
    if message.strip() == "":
        return "", history
        
    # Add user message to history immediately
    history.append({"role": "user", "content": message})
    
    # Process the response
    return asyncio.run(process_response(message, history))

def clear_history():
    """Clear both Gradio and chatbot history"""
    chatbot.chat_history = []
    return None, None, []

# Custom CSS for better markdown display
custom_css = """
footer {visibility: hidden}
.message-wrap {
    max-width: 100% !important;
}
.message.bot {
    width: 100% !important;
}
.message.bot .block.text-base {
    width: 100% !important;
}
pre {
    background-color: #f6f8fa !important;
    border-radius: 6px !important;
    padding: 16px !important;
    margin-top: 8px !important;
    margin-bottom: 8px !important;
}
code {
    background-color: rgba(175,184,193,0.2) !important;
    padding: 0.2em 0.4em !important;
    border-radius: 6px !important;
    font-family: ui-monospace,SFMono-Regular,SF Mono,Menlo,Consolas,Liberation Mono,monospace !important;
    font-size: 85% !important;
}
pre code {
    background-color: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
    border: 0 !important;
}
.dark pre {
    background-color: #1e1e1e !important;
}
.contains-code {
    width: 100% !important;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("""
    # HUST Chatbot
    Chào mừng bạn đến với trợ lý ảo của trường Đại học Bách khoa Hà Nội!
    """)
    
    chatbot_interface = gr.Chatbot(
        height=500,
        show_label=False,
        container=True,
        bubble_full_width=False,
        type="messages",
        rtl=False,  # Ensure left-to-right text direction
    )
    
    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Nhập câu hỏi của bạn...",
            container=False,
            lines=2  # Allow multiple lines input
        )
        submit_btn = gr.Button("Gửi", scale=1)
    
    clear_btn = gr.Button("Xóa lịch sử", variant="secondary")
    
    # Set up event handlers
    txt.submit(
        user_message,
        [txt, chatbot_interface],
        [txt, chatbot_interface]
    )
    
    submit_btn.click(
        user_message,
        [txt, chatbot_interface],
        [txt, chatbot_interface]
    )
    
    clear_btn.click(
        clear_history,
        None,
        [txt, clear_btn, chatbot_interface]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=False)