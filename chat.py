# chat.py 开头修改为：
from config import *
import json
import os
os.environ["MODELSCOPE_CACHE"] = MODELSCOPE_CACHE_DIR
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from create_kb import *

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
EMBED_MODEL = HuggingFaceEmbedding(
    model_name=EMBED_MODEL_PATH,
)
Settings.embed_model = EMBED_MODEL

# 加载本地 LLM
print("正在加载本地 LLM...")
llm = LlamaCPP(
    model_path=LLM_MODEL_PATH,
    temperature=0.1,          # 实际温度由前端 slider 控制，这里只是默认值
    max_new_tokens=512,       # 同上
    context_window=8192,      # 根据你的模型调整，14B 可设 8192 或 32768
    model_kwargs={"n_ctx": 8192, "verbose": False, "n_gpu_layers": 0},  # CPU 用 0
)
Settings.llm = llm
print("✅ 本地 LLM 加载成功")

# 全局单例加载重排模型，避免每次请求卡顿 1 分钟
RERANKER = SentenceTransformerRerank(
    model=RERANKER_MODEL_PATH,
    top_n=5
)

def chat_response(multi_modal_input, history, model, temperature, max_tokens, history_round, db_name, similarity_threshold, chunk_cnt):
    # ========== 提取 prompt（修正：从 multi_modal_input 提取） ==========
    prompt = ""
    if isinstance(multi_modal_input, dict) and multi_modal_input.get("text"):
        prompt = multi_modal_input["text"].strip()
    print(f"提取到的用户输入: {prompt}")  # 方便后续调试

    # ========== 临时文件处理（同原来） ==========
    tmp_files = multi_modal_input.get('files', [])
    if os.path.exists(os.path.join(FILE_ROOT_PATH, TMP_NAME)):
        db_name = TMP_NAME
    else:
        if tmp_files and len(tmp_files) > 0:
            create_tmp_kb(tmp_files)
            db_name = TMP_NAME

    print(f"prompt:{prompt}, tmp_files:{tmp_files}, db_name:{db_name}")

    # ========== RAG 检索（修正：空值拦截 + 防御性编程） ==========
    chunk_text = ""
    chunk_show = ""

    if not prompt:  # 如果输入为空，直接跳过 RAG 检索
        print("警告：用户输入为空，跳过 RAG 检索")
        final_prompt = "请输入您的问题内容。"
        chunk_show = ""
    else:
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=os.path.join(DB_PATH, db_name)
            )
            # 防止知识库不存在导致崩溃
            if not os.path.exists(os.path.join(DB_PATH, db_name)):
                raise FileNotFoundError(f"知识库 {db_name} 不存在")

            index = load_index_from_storage(storage_context)
            retriever_engine = index.as_retriever(similarity_top_k=20)
            retrieve_chunk = retriever_engine.retrieve(prompt)

            # ========== 本地重排序 ==========
            try:
                RERANKER.top_n = chunk_cnt
                results = RERANKER.postprocess_nodes(retrieve_chunk, query_str=prompt)
                print(f"重排序完成，返回 {len(results)} 个节点")
            except Exception as e:
                print(f"重排序失败，使用粗排结果: {e}")
                results = retrieve_chunk[:chunk_cnt]

            # 拼接召回内容（增加对 score 为 None 的防御 & 强制截断 & 内容去重）
            seen_texts = set()  # 【新增】用于存放已经添加过的文本内容
            for i, node in enumerate(results):
                if node.score is None:
                    node.score = 0.0
                if node.score >= similarity_threshold:
                    # 【新增】内容去重：如果这段文本已经被加入过，直接跳过，防止AI反复输出相同段落
                    clean_node_text = node.text.strip()
                    if clean_node_text in seen_texts:
                        continue
                    seen_texts.add(clean_node_text)

                    # 强制截断长文本防止溢出
                    safe_text = clean_node_text[:1500] + "..." if len(clean_node_text) > 1500 else clean_node_text
                    chunk_text += f"## {i + 1}: {safe_text}\n"
                    chunk_show += f"## {i + 1}: {safe_text}\nscore: {round(node.score, 2)}\n"

            # 构建最终 prompt
            if chunk_text.strip():
                final_prompt = f"""你是一个严谨的AI助手。请仅根据以下参考资料回答用户的问题。
如果资料中没有答案，请礼貌地表示无法回答，不要编造。
请直接给出答案内容，严禁输出“用户：”、“用户问题：”、“回答：”等前缀标签，严禁输出参考资料中与当前问题无关的其他内容。

参考资料：
{chunk_text}

用户问题：{prompt}

最终回答："""
            else:
                final_prompt = f"请回答用户的问题：{prompt}。如果不知道，请直接回答不知道。"
        except Exception as rag_e:
            print(f"RAG逻辑异常：{rag_e}")
            final_prompt = prompt
            chunk_show = ""

    # ========== 使用本地 LlamaCPP 生成回答（流式） ==========
    # 设置生成参数
    llm.temperature = temperature
    llm.max_new_tokens = max_tokens
    # 如果需要上下文历史，需把历史拼接进 prompt（LlamaCPP 不支持多轮 chat，需自己拼接）
    # 我们简单地将历史消息拼接在 final_prompt 前面
    context_prompt = ""
    if history:  # 如果有历史记录
        max_hist = history_round * 2
        # 修正：直接取最后 max_hist 条记录，不要用 [:-1] 切掉最后的 AI 回复
        recent_history = history[-max_hist:]
        for msg in recent_history:
            if msg["role"] == "user":
                context_prompt += f"用户：{msg['content']}\n"
            elif msg["role"] == "assistant":
                context_prompt += f"助手：{msg['content']}\n"
    full_prompt = context_prompt + f"用户：{final_prompt}\n助手："

    # 1. 防御性保护（防止 history 为 None）
    if history is None:
        history = []

    # 2. 保存用户提问到历史（空提问不记录，防止污染 UI）
    if prompt:
        history.append({"role": "user", "content": prompt})

    # 3. 添加空助手占位符（给下面的流式生成预留位置）
    # 【修正】直接追加助手消息，删除前面多余的 if not history: history = []
    history.append({"role": "assistant", "content": ""})

    # 流式生成
    try:
        response_gen = llm.stream_complete(full_prompt)
        full_ans = ""
        for response in response_gen:
            delta = response.delta
            if delta:
                full_ans += delta

                # 【扩充拦截词】把刚才发现的新格式也加进去
                stop_words = ["用户：", "用户问题：", "回答：", "助手：", "不知道\n"]
                if any(word in delta for word in stop_words) or any(word in full_ans for word in stop_words):
                    # 找到最先出现的停止词，一刀切
                    import re
                    full_ans = re.split(r"用户：|用户问题：|回答：|助手：|不知道", full_ans)[0].strip()
                    history[-1]["content"] = full_ans
                    yield history, chunk_show, {"text": "", "files": []}
                    break  # 强行停止生成

                history[-1]["content"] = full_ans
                yield history, chunk_show, {"text": "", "files": []}
        if not full_ans:
            history[-1]["content"] = "知识库里没有相关答案"
            yield history, chunk_show, {"text": "", "files": []}
    except Exception as e:
        history[-1]["content"] = f"本地模型推理失败：{str(e)}"
        yield history, chunk_show, {"text": "", "files": []}