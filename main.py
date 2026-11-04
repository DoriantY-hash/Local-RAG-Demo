from config import *
import sys
class FilterStderr:
    def __init__(self, real):
        self.real = real
    def write(self, message):
        if "cache_dir" not in message:
            self.real.write(message)
    def flush(self):
        self.real.flush()
sys.stderr = FilterStderr(sys.stderr)

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import os
os.environ['NLTK_INSECURE'] = '1'

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import gradio as gr
import os
os.environ["MODELSCOPE_CACHE"] = MODELSCOPE_CACHE_DIR
from html_string import main_html, plain_html
from upload_file import *
from create_kb import *  # 导入新版create_kb_demo
from chat import chat_response  # 导入修改后的函数

#####################################
######       gradio界面       #######
#####################################

def get_chat_block():
    with gr.Blocks() as chat:
        chat.theme = gr.themes.Base()
        chat.css = ".gradio_container { background-color: #f0f0f0; }"
        gr.HTML(plain_html)
        with gr.Row():     
            with gr.Column(scale=10):
                chatbot = gr.Chatbot(
                    label="Chatbot",
                    height=750,
                    value=[],
                    type="messages"
                )
                with gr.Row():
                    input_message = gr.MultimodalTextbox(
                        label="",
                        placeholder="在此输入消息...",
                        file_types=[".xlsx", ".csv", ".docx", ".pdf", ".txt"],
                        scale=7,
                        value={"text": "", "files": []}
                    )
                    clear_btn = gr.ClearButton(chatbot, input_message, scale=1)
            with gr.Column(scale=5):
                kb_choices = os.listdir(DB_PATH) if os.path.exists(DB_PATH) else []
                knowledge_base = gr.Dropdown(
                    choices=kb_choices,
                    label="加载知识库",
                    interactive=True,
                    scale=2
                )
                with gr.Accordion(label="召回文本段", open=False):
                    chunk_text = gr.Textbox(
                        label="召回文本段",
                        interactive=False,
                        scale=5,
                        lines=10
                    )
                with gr.Accordion(label="模型设置", open=True):
                    model = gr.Dropdown(
                        choices=['local-Qwen2.5-3B'],
                        label="选择模型",
                        interactive=True,
                        value="local-Qwen2.5-3B",
                        scale=2
                    )
                    temperature = gr.Slider(
                        maximum=2,
                        minimum=0,
                        interactive=True,
                        label="温度参数",
                        step=0.01,
                        value=0.2,
                        scale=2
                    )
                    max_tokens = gr.Slider(
                        maximum=8000,
                        minimum=0,
                        interactive=True,
                        label="最大回复长度",
                        step=50,
                        value=256,
                        scale=2
                    )
                    history_round = gr.Slider(
                        maximum=30,
                        minimum=1,
                        interactive=True,
                        label="携带上下文轮数",
                        step=1,
                        value=1,
                        scale=2
                    )
                with gr.Accordion(label="RAG参数设置", open=True):
                    chunk_cnt = gr.Slider(
                        maximum=20,
                        minimum=1,
                        interactive=True,
                        label="选择召回片段数",
                        step=1,
                        value=3,
                        scale=2
                    )
                    similarity_threshold = gr.Slider(
                        maximum=1,
                        minimum=0,
                        interactive=True,
                        label="相似度阈值",
                        step=0.01,
                        value=0.2,
                        scale=2
                    )
        
        # 修改事件绑定：直接调用 chat_response，接收原始输入并输出清空的输入框
        input_message.submit(
            fn=chat_response,
            inputs=[
                input_message, chatbot, model, temperature,
                max_tokens, history_round, knowledge_base,
                similarity_threshold, chunk_cnt
            ],
            outputs=[chatbot, chunk_text, input_message],  # 注意 output 包括 input_message 用于清空
            queue=True
        )
        
        # 页面加载时刷新知识库列表，并清除临时知识库（问题1修复）
        chat.load(update_knowledge_base, [], knowledge_base)
        # 加载时清除临时文件和知识库
        chat.load(fn=clear_tmp, inputs=None, outputs=None)
    return chat

def get_upload_block():
    with gr.Blocks() as upload:
        upload.theme = gr.themes.Base()
        gr.HTML(plain_html)
        with gr.Tab("非结构化数据"):
            with gr.Accordion(label="新建类目", open=True):
                with gr.Column(scale=2):
                    unstructured_file = gr.Files(file_types=[".pdf", ".docx", ".txt"])
                    with gr.Row():
                        new_label = gr.Textbox(label="类目名称", placeholder="请输入类目名称", scale=5)
                        create_label_btn = gr.Button("新建类目", variant="primary", scale=1)
            with gr.Accordion(label="管理类目", open=False):
                with gr.Row():
                    unstructured_choices = os.listdir(UNSTRUCTURED_FILE_PATH) if os.path.exists(UNSTRUCTURED_FILE_PATH) else []
                    data_label = gr.Dropdown(
                        choices=unstructured_choices,
                        label="管理类目",
                        interactive=True,
                        scale=8,
                        multiselect=True
                    )
                    delete_label_btn = gr.Button("删除类目", variant="stop", scale=1)
        with gr.Tab("结构化数据"):
            with gr.Accordion(label="新建数据表", open=True):
                with gr.Column(scale=2):
                    structured_file = gr.Files(file_types=[".xlsx", ".csv"])
                    with gr.Row():
                        new_label_1 = gr.Textbox(label="数据表名称", placeholder="请输入数据表名称", scale=5)
                        create_label_btn_1 = gr.Button("新建数据表", variant="primary", scale=1)
            with gr.Accordion(label="管理数据表", open=False):
                with gr.Row():
                    structured_choices = os.listdir(STRUCTURED_FILE_PATH) if os.path.exists(STRUCTURED_FILE_PATH) else []
                    data_label_1 = gr.Dropdown(
                        choices=structured_choices,
                        label="管理数据表",
                        interactive=True,
                        scale=8,
                        multiselect=True
                    )
                    delete_data_table_btn = gr.Button("删除数据表", variant="stop", scale=1)
        delete_label_btn.click(delete_label, inputs=[data_label]).then(fn=update_label, outputs=[data_label])
        create_label_btn.click(fn=upload_unstructured_file, inputs=[unstructured_file, new_label]).then(fn=update_label, outputs=[data_label])
        delete_data_table_btn.click(delete_data_table, inputs=[data_label_1]).then(fn=update_datatable, outputs=[data_label_1])
        create_label_btn_1.click(fn=upload_structured_file, inputs=[structured_file, new_label_1]).then(fn=update_datatable, outputs=[data_label_1])
        upload.load(update_label, [], data_label)
        upload.load(update_datatable, [], data_label_1)
    return upload

def get_knowledge_base_block():
    return create_kb_demo()

app = FastAPI()
@app.get("/", response_class=HTMLResponse)
def read_main():
    html_content = main_html
    return HTMLResponse(content=html_content)

app = gr.mount_gradio_app(app, get_chat_block(), path="/chat")
app = gr.mount_gradio_app(app, get_upload_block(), path="/upload_data")
app = gr.mount_gradio_app(app, get_knowledge_base_block(), path="/create_knowledge_base")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)