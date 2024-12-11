import gradio as gr
from typing import List, Dict
import asyncio
from dotenv import load_dotenv
from core.chatbot import HustChatbot
from router.router import LangchainRouter
from rag.retrieval import SupabaseRetriever
from rag.responder import RAGResponder
from reflection.reflection import LangchainReflection
from rag.responder import ChitchatResponder
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.3
)

router = LangchainRouter(llm=llm)
retriever = SupabaseRetriever()
reflection = LangchainReflection(llm=llm)
chitchat_responder = ChitchatResponder(llm=llm)
rag_responder = RAGResponder(llm=llm)

chatbot = HustChatbot(
    router=router,
    retriever=retriever,
    reflection=reflection,
    chitchat_responder=chitchat_responder,
    rag_responder=rag_responder
)

async def stream_response(message: str, history: List[Dict]) :
    if message.strip() == "":
        yield history
        return

    history = history + [{"role": "user", "content": message}]
    response = ""
    
    try:
        async for chunk in chatbot.stream_process_message(message):
            response += chunk
            if len(history) > 0 and history[-1]["role"] == "assistant":
                history[-1]["content"] = response
            else:
                history.append({"role": "assistant", "content": response})
            yield history, ""
    except Exception as e:
        error_msg = f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
        if len(history) > 0 and history[-1]["role"] == "assistant":
            history[-1]["content"] = error_msg
        else:
            history.append({"role": "assistant", "content": error_msg})
        yield history, ""

def clear_history():
    """Clear both Gradio and chatbot history"""
    chatbot.chat_history = []
    chatbot.prev_question = ""
    return None, None, []

# Custom CSS for better styling
custom_css = """
    footer {display: none !important;}
    .gradio-container {
        min-height: 0px !important;
        max-width: 1200px !important;
        margin: auto !important;
    }
    .header-container {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header-content {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .header-text h1 {
        color: #2d2d2d;
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .header-text p {
        color: #666;
        font-size: 1.1rem;
    }
    .chat-container {
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .input-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0 0 10px 10px;
    }
    .submit-btn {
        background-color: #007bff !important;
        color: white !important;
    }
    .clear-btn {
        background-color: #6c757d !important;
        color: white !important;
    }
"""

# Create Blocks with custom title and styling
demo = gr.Blocks(
    title="HUST Assistant",
    css=custom_css
)

with demo:
    # Header section with improved styling
    gr.Markdown("""
        <div class="header-container">
            <div class="header-content">
                <img src="https://www.cio.com/wp-content/uploads/2023/08/chatbot_ai_machine-learning_emerging-tech-100778305-orig.jpg?resize=1024%2C683&quality=50&strip=all" width="100" height="100" style="border-radius: 50%;">
                <div class="header-text">
                    <h1>HUST Assistant</h1>
                    <p>Chào mừng bạn đến với trợ lý ảo của trường Đại học Bách khoa Hà Nội!</p>
                    <p style="font-size: 0.9rem; color: #666; margin-top: 0.5rem;">
                        💡 Hãy đặt câu hỏi về thông tin tuyển sinh, cẩm nang sinh viên, hoặc các vấn đề khác liên quan đến trường.
                    </p>
                </div>
            </div>
        </div>
    """)
    
    with gr.Column(elem_classes="chat-container"):
        chatbot_interface = gr.Chatbot(
            height=500,
            show_label=False,
            container=True,
            bubble_full_width=False,
            show_copy_button=True,
            render_markdown=True,
            type="messages",
            elem_classes="chat-messages"
        )
        
        with gr.Column(elem_classes="input-container"):
            with gr.Row():
                txt = gr.Textbox(
                    scale=4,
                    show_label=False,
                    placeholder="Nhập câu hỏi của bạn... (Nhấn Enter để gửi)",
                    container=False,
                    lines=2,
                    autofocus=True,
                    elem_classes="chat-input",
                )
                submit_btn = gr.Button("Gửi", scale=1, elem_classes="submit-btn")
            
            with gr.Row():
                clear_btn = gr.Button(
                    "🗑️ Xóa lịch sử", 
                    variant="secondary",
                    elem_classes="clear-btn"
                )
    
    # Event handlers
    txt.submit(
        stream_response, 
        [txt, chatbot_interface], 
        [chatbot_interface, txt], 
        queue=True
    )

    submit_btn.click(
        stream_response, 
        [txt, chatbot_interface], 
        [chatbot_interface, txt], 
        queue=True
    )

    clear_btn.click(clear_history, None, [txt, clear_btn, chatbot_interface])

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        share=False,
    )