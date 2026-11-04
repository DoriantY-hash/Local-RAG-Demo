#####################################
######       创建知识库         #######
#####################################
from config import *
import gradio as gr
import os
import shutil
import traceback
import time
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode, Document
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from html_string import plain_html

# 确保基础目录存在
for path in [DB_PATH, STRUCTURED_FILE_PATH, UNSTRUCTURED_FILE_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

# 本地嵌入模型配置
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

try:
    EMBED_MODEL = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_PATH,
    )
    Settings.embed_model = EMBED_MODEL
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    print("嵌入模型加载成功")
except Exception as e:
    print(f"嵌入模型加载失败: {e}")
    raise

def load_existing_index(db_name: str):
    """加载已有知识库的向量索引"""
    db_path = os.path.join(DB_PATH, db_name)
    if not os.path.exists(db_path):
        return None, None
    try:
        storage_context = StorageContext.from_defaults(persist_dir=db_path)
        index = load_index_from_storage(storage_context)
        docstore = storage_context.docstore
        return index, docstore
    except Exception as e:
        print(f"加载索引失败：{e}")
        return None, None

def refresh_knowledge_base():
    return [name for name in os.listdir(DB_PATH) if os.path.isdir(os.path.join(DB_PATH, name))]

def get_label_list(is_structured: bool):
    """获取仅包含文件的结构化/非结构化一级目录，空文件夹自动排除"""
    target_path = STRUCTURED_FILE_PATH if is_structured else UNSTRUCTURED_FILE_PATH
    if not os.path.exists(target_path):
        return []
    valid_labels = []
    for folder_name in os.listdir(target_path):
        folder_path = os.path.join(target_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if os.listdir(folder_path):
            valid_labels.append(folder_name)
    return valid_labels

def clean_empty_folders(is_structured: bool):
    """遍历目录，删除所有空文件夹"""
    target_path = STRUCTURED_FILE_PATH if is_structured else UNSTRUCTURED_FILE_PATH
    if not os.path.exists(target_path):
        return
    for folder_name in os.listdir(target_path):
        folder_path = os.path.join(target_path, folder_name)
        if os.path.isdir(folder_path) and len(os.listdir(folder_path)) == 0:
            try:
                os.rmdir(folder_path)
                print(f"已自动删除空目录: {folder_path}")
            except:
                pass

def get_kb_files(db_name: str, is_structured: bool = False):
    """获取知识库关联的文件列表"""
    file_path = STRUCTURED_FILE_PATH if is_structured else UNSTRUCTURED_FILE_PATH
    kb_files = []
    if os.path.exists(file_path):
        for label in os.listdir(file_path):
            label_path = os.path.join(file_path, label)
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    kb_files.append(os.path.join(label, file))
    return kb_files

# ========== 修改：删除文件并更新知识库，返回更新后的下拉框 ==========
def delete_files_and_update_kb(db_name: str, files_to_delete: list, is_structured: bool = False, progress=gr.Progress(track_tqdm=True)):
    """删除指定文件并更新知识库，返回更新后的知识库下拉框和文件列表"""
    if not db_name:
        gr.Info("请选择要更新的知识库")
        return gr.update(), gr.update()
    if not files_to_delete:
        gr.Info("请选择要删除的文件")
        return gr.update(), gr.update()

    file_root = STRUCTURED_FILE_PATH if is_structured else UNSTRUCTURED_FILE_PATH
    deleted_files = []
    for file_rel_path in files_to_delete:
        file_abs_path = os.path.join(file_root, file_rel_path)
        if os.path.exists(file_abs_path):
            try:
                os.remove(file_abs_path)
                deleted_files.append(file_rel_path)
                print(f"已删除文件: {file_abs_path}")
            except Exception as e:
                gr.Warning(f"删除文件 {file_rel_path} 失败: {e}")

    if not deleted_files:
        gr.Info("没有成功删除任何文件")
        return gr.update(), gr.update()

    clean_empty_folders(is_structured)

    gr.Info(f"正在重建 [{db_name}] 知识库索引，请等待...")
    try:
        if is_structured:
            all_tables = get_label_list(is_structured=True)
            documents = []
            parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
            for label in progress.tqdm(all_tables, desc="加载数据表"):
                label_path = os.path.join(STRUCTURED_FILE_PATH, label)
                if os.path.exists(label_path):
                    docs = SimpleDirectoryReader(label_path).load_data()
                    # 【重点替换】同样调用解析器生成切好的小块
                    nodes_from_doc = parser.get_nodes_from_documents(docs)
                    documents.extend(nodes_from_doc)
        else:
            all_labels = get_label_list(is_structured=False)
            documents = []
            parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)  # 引入切分器
            for label in progress.tqdm(all_labels, desc="加载类目"):
                label_path = os.path.join(UNSTRUCTURED_FILE_PATH, label)
                if os.path.exists(label_path):
                    docs = SimpleDirectoryReader(label_path).load_data()
                    # 【重点修复】使用 get_nodes_from_documents 生成切好的小块
                    nodes_from_doc = parser.get_nodes_from_documents(docs)
                    documents.extend(nodes_from_doc)

        if not documents:
            db_path = os.path.join(DB_PATH, db_name)
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
            gr.Info(f"知识库 [{db_name}] 已无文档，已删除该知识库")
            return gr.update(choices=refresh_knowledge_base()), gr.update(choices=[])

        index = VectorStoreIndex(documents, show_progress=True)
        db_path = os.path.join(DB_PATH, db_name)
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        index.storage_context.persist(db_path)
        progress(1.0, desc="保存完成")
        gr.Info(f"[{db_name}] 知识库更新完成（已删除指定文件并重建索引）")
    except Exception as e:
        gr.Error(f"更新知识库失败: {e}")
        print(f"更新知识库异常: {e}")
        traceback.print_exc()
    # 返回更新后的知识库下拉框选项和清空的文件列表
    return gr.update(choices=refresh_knowledge_base()), gr.update(choices=[])

# ========== 修改：创建非结构化知识库，返回更新后的下拉框 ==========
def create_unstructured_db(db_name: str, label_name: list, is_update, progress=gr.Progress(track_tqdm=True)):
    is_update = (is_update == "增量更新已有知识库")
    print(f"知识库名称：{db_name}，类目：{label_name}，是否更新：{is_update}")
    if label_name is None or len(label_name) == 0:
        gr.Info("没有选择类目")
        return gr.update(), gr.update()
    if len(db_name) == 0:
        gr.Info("没有命名知识库")
        return gr.update(), gr.update()

    if is_update:
        if db_name not in refresh_knowledge_base():
            gr.Info(f"[{db_name}] 知识库不存在，无法更新")
            return gr.update(), gr.update()
        gr.Info(f"正在增量更新 [{db_name}] 知识库，请等待...")
        existing_index, _ = load_existing_index(db_name)
        if existing_index is None:
            gr.Info(f"[{db_name}] 知识库索引加载失败，将执行全量重建")
            is_update = False

        if is_update:
            new_nodes = []
            parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
            for label in progress.tqdm(label_name, desc="加载新文件"):
                label_path = os.path.join(UNSTRUCTURED_FILE_PATH, label)
                if os.path.exists(label_path):
                    try:
                        docs = SimpleDirectoryReader(label_path).load_data()
                        # 【重点修改】增量更新时，新上传的文档也要切分成小节点再插入
                        nodes = parser.get_nodes_from_documents(docs)
                        new_nodes.extend(nodes)
                        print(f"类目 {label} 加载文件数: {len(docs)}，拆分后新增节点数: {len(nodes)}")
                    except Exception as e:
                        gr.Warning(f"读取类目 {label} 文件失败: {e}")
                        print(f"读取类目 {label} 文件失败: {e}")
                else:
                    gr.Info(f"类目 {label} 路径不存在，跳过")

            if len(new_nodes) == 0:
                gr.Info("没有加载到新文件，更新终止")
                return gr.update(), gr.update()

            try:
                progress(0.5, desc="插入节点")
                existing_index.insert_nodes(new_nodes)
                progress(0.8, desc="保存索引")
                existing_index.storage_context.persist(os.path.join(DB_PATH, db_name))
                progress(1.0, desc="完成")
                gr.Info(f"[{db_name}] 非结构化知识库增量更新成功，可前往RAG问答提问")
                return gr.update(choices=refresh_knowledge_base()), gr.update()
            except Exception as e:
                gr.Error(f"增量更新失败: {e}")
                print(f"增量更新异常: {e}")
                traceback.print_exc()
                return gr.update(), gr.update()

    if db_name in refresh_knowledge_base() and not is_update:
        gr.Info("知识库已存在，请换个名字或删除原来知识库再创建")
        return gr.update(), gr.update()

    gr.Info("正在创建/重建知识库，请等待完成后前往RAG问答")
    nodes = []
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    for label in progress.tqdm(label_name, desc="加载类目"):
        label_path = os.path.join(UNSTRUCTURED_FILE_PATH, label)
        if os.path.exists(label_path):
            try:
                docs = SimpleDirectoryReader(label_path).load_data()
                # 【重点修改】使用解析器切分，替代原来通篇保存为1个节点的方法
                nodes.extend(parser.get_nodes_from_documents(docs))
                print(f"类目 {label} 加载文件数: {len(docs)}，拆分后生成节点数: {len(nodes)}")
            except Exception as e:
                gr.Warning(f"读取类目 {label} 文件失败: {e}")
                print(f"读取类目 {label} 文件失败: {e}")
        else:
            gr.Info(f"类目 {label} 路径不存在，跳过")

    if len(nodes) == 0:
        gr.Info("没有加载到任何有效文件，创建/重建终止")
        print("错误：没有加载到任何有效文件")
        return gr.update(), gr.update()

    try:
        print("开始构建向量索引...")
        index = VectorStoreIndex(nodes, show_progress=True)
        print("向量索引构建完成，正在保存...")
        db_path = os.path.join(DB_PATH, db_name)
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        index.storage_context.persist(db_path)
        progress(1.0, desc="保存完成")
        print(f"知识库已保存至: {db_path}")
        gr.Info("非结构化知识库创建/重建成功，可前往RAG问答进行提问")
    except Exception as e:
        gr.Error(f"创建知识库失败: {e}")
        print(f"创建知识库异常: {e}")
        traceback.print_exc()
    return gr.update(choices=refresh_knowledge_base()), gr.update()

# ========== 修改：创建结构化知识库，返回更新后的下拉框 ==========
def create_structured_db(db_name: str, data_table: list, is_update, progress=gr.Progress(track_tqdm=True)):
    is_update = (is_update == "增量更新已有知识库")
    print(f"知识库名称：{db_name}，数据表：{data_table}，是否更新：{is_update}")
    if data_table is None or len(data_table) == 0:
        gr.Info("没有选择数据表")
        return gr.update(), gr.update()
    if len(db_name) == 0:
        gr.Info("没有命名知识库")
        return gr.update(), gr.update()

    if is_update:
        if db_name not in refresh_knowledge_base():
            gr.Info(f"[{db_name}] 知识库不存在，无法更新")
            return gr.update(), gr.update()
        gr.Info(f"正在增量更新 [{db_name}] 知识库，请等待...")
        existing_index, _ = load_existing_index(db_name)
        if existing_index is None:
            gr.Info(f"[{db_name}] 知识库索引加载失败，将执行全量重建")
            is_update = False

        if is_update:
            new_nodes = []
            parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
            for label in progress.tqdm(data_table, desc="加载数据表"):
                label_path = os.path.join(STRUCTURED_FILE_PATH, label)
                if os.path.exists(label_path):
                    try:
                        docs = SimpleDirectoryReader(label_path).load_data()
                        print(f"数据表 {label} 加载文件数: {len(docs)}")
                        # 【重点修改】同样使用解析器切分小块
                        nodes = parser.get_nodes_from_documents(docs)
                        new_nodes.extend(nodes)
                        print(f"数据表 {label} 生成节点数: {len(new_nodes)}")
                    except Exception as e:
                        gr.Warning(f"读取数据表 {label} 文件失败: {e}")
                        print(f"读取数据表 {label} 文件失败: {e}")
                else:
                    gr.Info(f"数据表 {label} 路径不存在，跳过")

            if len(new_nodes) == 0:
                gr.Info("没有加载到新数据，更新终止")
                return gr.update(), gr.update()

            try:
                progress(0.5, desc="插入节点")
                existing_index.insert_nodes(new_nodes)
                progress(0.8, desc="保存索引")
                existing_index.storage_context.persist(os.path.join(DB_PATH, db_name))
                progress(1.0, desc="完成")
                gr.Info(f"[{db_name}] 结构化知识库增量更新成功，可前往RAG问答提问")
                return gr.update(choices=refresh_knowledge_base()), gr.update()
            except Exception as e:
                gr.Error(f"增量更新失败: {e}")
                print(f"增量更新异常: {e}")
                traceback.print_exc()
                return gr.update(), gr.update()

    if db_name in refresh_knowledge_base() and not is_update:
        gr.Info("知识库已存在，请换个名字或删除原来知识库再创建")
        return gr.update(), gr.update()

    gr.Info("正在创建/重建知识库，请等待完成后前往RAG问答")
    documents = []
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    for label in progress.tqdm(data_table, desc="加载数据表"):
        label_path = os.path.join(STRUCTURED_FILE_PATH, label)
        if os.path.exists(label_path):
            print(f"正在读取数据表目录: {label_path}")
            try:
                docs = SimpleDirectoryReader(label_path).load_data()
                print(f"数据表 {label} 加载文件数: {len(docs)}")
                # 【重点替换】统一使用解析器切分
                nodes = parser.get_nodes_from_documents(docs)
                documents.extend(nodes)
                print(f"数据表 {label} 已生成节点数: {len(nodes)}")
            except Exception as e:
                gr.Warning(f"读取数据表 {label} 文件失败: {e}")
                print(f"读取数据表 {label} 文件失败: {e}")
                traceback.print_exc()
        else:
            gr.Info(f"数据表 {label} 路径不存在，跳过")

    if len(documents) == 0:
        gr.Info("没有加载到任何有效数据，创建/重建终止")
        print("错误：没有加载到任何有效数据，请检查数据表目录是否包含有效文件且文件内容不为空")
        return gr.update(), gr.update()

    try:
        print("测试嵌入模型是否可用...")
        test_text = "这是一个测试文本"
        test_embedding = Settings.embed_model.get_text_embedding(test_text)
        print(f"嵌入模型测试成功，向量维度: {len(test_embedding)}")
    except Exception as e:
        gr.Error(f"嵌入模型测试失败: {e}")
        print(f"嵌入模型测试失败: {e}")
        traceback.print_exc()
        return gr.update(), gr.update()

    start_time = time.time()
    try:
        print(f"开始构建向量索引，文档数量: {len(documents)}，时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        gr.Info(f"正在构建向量索引，文档数量: {len(documents)}，请耐心等待...")
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        elapsed = time.time() - start_time
        print(f"向量索引构建完成，耗时: {elapsed:.2f} 秒，时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        print("正在保存知识库...")
        db_path = os.path.join(DB_PATH, db_name)
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        index.storage_context.persist(db_path)
        progress(1.0, desc="保存完成")
        print(f"知识库已保存至: {db_path}")
        gr.Info("结构化知识库创建/重建成功，可前往RAG问答进行提问")
    except Exception as e:
        gr.Error(f"创建知识库失败: {e}")
        print(f"创建知识库异常: {e}")
        traceback.print_exc()
    return gr.update(choices=refresh_knowledge_base()), gr.update()

def delete_db(db_name: str):
    """删除知识库，并返回更新后的下拉框"""
    if db_name is None or db_name.strip() == "":
        gr.Info("请选择要删除的知识库")
        return gr.update()
    folder_path = os.path.join(DB_PATH, db_name)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        gr.Info(f"已成功删除{db_name}知识库")
        print(f"已成功删除{db_name}知识库")
    else:
        gr.Info(f"{db_name}知识库不存在")
        print(f"{db_name}知识库不存在")
    return gr.update(choices=refresh_knowledge_base())

def update_knowledge_base():
    return gr.update(choices=refresh_knowledge_base())

def refresh_file_list(db_name: str, is_structured: bool):
    if not db_name:
        return gr.update(choices=[])
    files = get_kb_files(db_name, is_structured)
    return gr.update(choices=files)

def refresh_label_list(is_structured: bool):
    labels = get_label_list(is_structured)
    return gr.update(choices=labels)

def create_tmp_kb(files, progress=gr.Progress(track_tqdm=True)):
    tmp_file_path = os.path.join(FILE_ROOT_PATH, TMP_NAME)
    if not os.path.exists(tmp_file_path):
        os.makedirs(tmp_file_path)

    valid_files = []
    for file in files:
        if os.path.exists(file):
            file_name = os.path.basename(file)
            target_path = os.path.join(tmp_file_path, file_name)
            shutil.copy2(file, target_path)
            valid_files.append(target_path)

    if not valid_files:
        gr.Info("没有有效文件可创建临时知识库")
        return

    progress(0.3, desc="复制文件完成")
    documents = SimpleDirectoryReader(tmp_file_path).load_data()
    valid_docs = [doc for doc in documents if doc.get_content().strip()]
    if not valid_docs:
        gr.Info("临时文件内容为空，创建终止")
        return

    progress(0.6, desc="构建索引")
    index = VectorStoreIndex.from_documents(valid_docs, show_progress=True)
    db_path = os.path.join(DB_PATH, TMP_NAME)
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    index.storage_context.persist(db_path)
    progress(1.0, desc="保存完成")
    gr.Info("临时知识库创建成功")

def clear_tmp():
    tmp_file_path = os.path.join(FILE_ROOT_PATH, TMP_NAME)
    tmp_db_path = os.path.join(DB_PATH, TMP_NAME)
    if os.path.exists(tmp_file_path):
        shutil.rmtree(tmp_file_path)
    if os.path.exists(tmp_db_path):
        shutil.rmtree(tmp_db_path)
    gr.Info("临时文件和临时知识库已清除")

def create_kb_demo():
    with gr.Blocks(title="创建/更新知识库") as demo:
        gr.HTML(plain_html)
        gr.Markdown("## 知识库管理系统")
        gr.Markdown("支持创建、增量更新、删除文件更新、全量重建、删除知识库等操作")

        # 非结构化知识库标签页
        with gr.Tab("非结构化知识库"):
            with gr.Row():
                with gr.Column(scale=2):
                    db_name_unstruct = gr.Textbox(
                        label="知识库名称",
                        placeholder="输入要创建/更新的知识库名称",
                        max_lines=1
                    )
                    with gr.Row():
                        label_list = gr.CheckboxGroup(
                            choices=get_label_list(is_structured=False),
                            label="选择文件类目"
                        )
                        refresh_label_unstruct = gr.Button("刷新类目列表", size="sm")
                    operate_type_unstruct = gr.Radio(
                        choices=["新建知识库", "增量更新已有知识库"],
                        label="操作类型",
                        value="新建知识库"
                    )
                    create_unstruct_btn = gr.Button("执行创建/增量更新", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### 删除文件并更新")
                    kb_select_unstruct = gr.Dropdown(
                        choices=refresh_knowledge_base(),
                        label="选择知识库",
                        value=None
                    )
                    file_list_unstruct = gr.CheckboxGroup(
                        choices=[],
                        label="选择要删除的文件"
                    )
                    with gr.Row():
                        refresh_files_unstruct = gr.Button("刷新文件列表")
                        delete_file_unstruct = gr.Button("删除文件并更新", variant="stop")

            # 绑定事件并更新下拉框
            create_unstruct_btn.click(
                fn=create_unstructured_db,
                inputs=[db_name_unstruct, label_list, operate_type_unstruct],
                outputs=[kb_select_unstruct, file_list_unstruct]  # 更新知识库下拉框和文件列表
            )
            refresh_label_unstruct.click(
                fn=lambda: refresh_label_list(is_structured=False),
                inputs=[],
                outputs=[label_list]
            )
            refresh_files_unstruct.click(
                fn=lambda kb: refresh_file_list(kb, is_structured=False),
                inputs=[kb_select_unstruct],
                outputs=[file_list_unstruct]
            )
            delete_file_unstruct.click(
                fn=delete_files_and_update_kb,
                inputs=[kb_select_unstruct, file_list_unstruct, gr.State(False)],  # is_structured=False
                outputs=[kb_select_unstruct, file_list_unstruct]  # 更新知识库下拉框和文件列表
            )

        # 结构化知识库标签页
        with gr.Tab("结构化知识库"):
            with gr.Row():
                with gr.Column(scale=2):
                    db_name_struct = gr.Textbox(
                        label="知识库名称",
                        placeholder="输入要创建/更新的知识库名称",
                        max_lines=1
                    )
                    with gr.Row():
                        table_list = gr.CheckboxGroup(
                            choices=get_label_list(is_structured=True),
                            label="选择数据表"
                        )
                        refresh_label_struct = gr.Button("刷新数据表列表", size="sm")
                    operate_type_struct = gr.Radio(
                        choices=["新建知识库", "增量更新已有知识库"],
                        label="操作类型",
                        value="新建知识库"
                    )
                    create_struct_btn = gr.Button("执行创建/增量更新", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### 删除文件并更新")
                    kb_select_struct = gr.Dropdown(
                        choices=refresh_knowledge_base(),
                        label="选择知识库",
                        value=None
                    )
                    file_list_struct = gr.CheckboxGroup(
                        choices=[],
                        label="选择要删除的文件"
                    )
                    with gr.Row():
                        refresh_files_struct = gr.Button("刷新文件列表")
                        delete_file_struct = gr.Button("删除文件并更新", variant="stop")

            create_struct_btn.click(
                fn=create_structured_db,
                inputs=[db_name_struct, table_list, operate_type_struct],
                outputs=[kb_select_struct, file_list_struct]
            )
            refresh_label_struct.click(
                fn=lambda: refresh_label_list(is_structured=True),
                inputs=[],
                outputs=[table_list]
            )
            refresh_files_struct.click(
                fn=lambda kb: refresh_file_list(kb, is_structured=True),
                inputs=[kb_select_struct],
                outputs=[file_list_struct]
            )
            delete_file_struct.click(
                fn=delete_files_and_update_kb,
                inputs=[kb_select_struct, file_list_struct, gr.State(True)],  # is_structured=True
                outputs=[kb_select_struct, file_list_struct]
            )

        # 删除知识库标签页
        with gr.Tab("删除知识库"):
            kb_list = gr.Dropdown(
                choices=refresh_knowledge_base(),
                label="选择要删除的知识库"
            )
            with gr.Row():
                delete_btn = gr.Button("删除知识库", variant="stop")
                refresh_btn = gr.Button("刷新知识库列表")
            delete_btn.click(
                fn=delete_db,
                inputs=[kb_list],
                outputs=kb_list  # 删除后更新下拉框
            )
            refresh_btn.click(
                fn=update_knowledge_base,
                inputs=None,
                outputs=kb_list
            )

        # 临时知识库管理
        with gr.Tab("临时知识库"):
            gr.Markdown("### 临时文件快速创建知识库（页面刷新后失效）")
            file_upload = gr.File(
                label="上传文件",
                file_count="multiple",
                file_types=[".txt", ".pdf", ".docx", ".md"]
            )
            with gr.Row():
                create_tmp_btn = gr.Button("创建临时知识库", variant="primary")
                clear_tmp_btn = gr.Button("清除临时数据", variant="stop")

            create_tmp_btn.click(
                fn=create_tmp_kb,
                inputs=[file_upload],
                outputs=None
            )
            clear_tmp_btn.click(
                fn=clear_tmp,
                inputs=None,
                outputs=None
            )

    return demo

if __name__ == "__main__":
    demo = create_kb_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True
    )