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
    temperature=0
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

# Create Blocks with custom title and favicon
demo = gr.Blocks(
    title="HUST",  # Custom tab title
    css="""
        footer {display: none !important;}
        .gradio-container {min-height: 0px !important;}
    """  # Hide Gradio footer and adjust container
)

with demo:
    
    gr.Markdown("""
        <div style="display: flex; align-items: center; gap: 20px;">
            <img src="https://www.hust.edu.vn/uploads/sys/logo-dhbk-1.png" width="80" height="80">
            <div>
                <h1 style="margin: 0;">HUST Assistant</h1>
                <p style="margin: 0;">Chào mừng bạn đến với trợ lý ảo của trường Đại học Bách khoa Hà Nội!</p>
            </div>
        </div>
    """)
    
    chatbot_interface = gr.Chatbot(
        height=500,
        show_label=False,
        container=True,
        bubble_full_width=False,
        show_copy_button=True,
        render_markdown=True,
        type="messages"
    )
    
    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Nhập câu hỏi của bạn...",
            container=False,
            lines=1,
            autofocus=True
        )
        submit_btn = gr.Button("Gửi", scale=1)
    
    clear_btn = gr.Button("Xóa lịch sử", variant="secondary")
    
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