from config import *
import os
import shutil
import gradio as gr
from pathlib import Path

# 确保基础目录存在
for path in [STRUCTURED_FILE_PATH, UNSTRUCTURED_FILE_PATH]:
    os.makedirs(path, exist_ok=True)

def upload_unstructured_file(files, label_name):
    """上传非结构化文件到指定类目目录，直接保存原文件"""
    if not files:
        gr.Info("请选择文件")
        return
    if not label_name or label_name.strip() == "":
        gr.Info("请输入类目名称")
        return

    target_dir = os.path.join(UNSTRUCTURED_FILE_PATH, label_name.strip())
    os.makedirs(target_dir, exist_ok=True)

    for file in files:
        # 假设 file 是字典，包含 name 和 path
        if isinstance(file, dict):
            src_path = file['path']
            file_name = file['name']
        else:
            src_path = file
            file_name = os.path.basename(src_path)
        dst_path = os.path.join(target_dir, file_name)
        shutil.copy2(src_path, dst_path)

    gr.Info(f"成功上传 {len(files)} 个文件到类目 {label_name}")
    # 返回更新后的类目列表（用于刷新下拉框）
    return gr.update(choices=get_label_list(is_structured=False))

def upload_structured_file(files, table_name):
    """上传结构化文件（Excel/CSV）到指定数据表目录，只保存原始文件，不生成额外 txt"""
    if not files:
        gr.Info("请选择文件")
        return
    if not table_name or table_name.strip() == "":
        gr.Info("请输入数据表名称")
        return

    target_dir = os.path.join(STRUCTURED_FILE_PATH, table_name.strip())
    os.makedirs(target_dir, exist_ok=True)

    for file in files:
        if isinstance(file, dict):
            src_path = file['path']
            file_name = file['name']
        else:
            src_path = file
            file_name = os.path.basename(src_path)

        # 只允许 .xlsx 和 .csv 文件
        if not (file_name.endswith('.xlsx') or file_name.endswith('.csv')):
            gr.Warning(f"文件 {file_name} 不是 Excel 或 CSV 格式，已跳过")
            continue

        dst_path = os.path.join(target_dir, file_name)
        shutil.copy2(src_path, dst_path)
        # 不再生成 txt 文件

    gr.Info(f"成功上传 {len(files)} 个文件到数据表 {table_name}")
    return gr.update(choices=get_label_list(is_structured=True))

def get_label_list(is_structured: bool):
    """获取仅包含文件的目录列表（空目录自动过滤）"""
    target_path = STRUCTURED_FILE_PATH if is_structured else UNSTRUCTURED_FILE_PATH
    if not os.path.exists(target_path):
        return []
    valid_labels = []
    for folder_name in os.listdir(target_path):
        folder_path = os.path.join(target_path, folder_name)
        if os.path.isdir(folder_path) and os.listdir(folder_path):
            valid_labels.append(folder_name)
    return valid_labels

def delete_label(labels):
    """删除非结构化类目及其所有文件"""
    if not labels:
        gr.Info("请选择要删除的类目")
        return gr.update(choices=get_label_list(is_structured=False))
    for label in labels:
        folder_path = os.path.join(UNSTRUCTURED_FILE_PATH, label)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
    gr.Info(f"已删除类目: {', '.join(labels)}")
    return gr.update(choices=get_label_list(is_structured=False))

def delete_data_table(tables):
    """删除结构化数据表及其所有文件"""
    if not tables:
        gr.Info("请选择要删除的数据表")
        return gr.update(choices=get_label_list(is_structured=True))
    for table in tables:
        folder_path = os.path.join(STRUCTURED_FILE_PATH, table)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
    gr.Info(f"已删除数据表: {', '.join(tables)}")
    return gr.update(choices=get_label_list(is_structured=True))

def update_label():
    """更新非结构化类目下拉框选项"""
    return gr.update(choices=get_label_list(is_structured=False))

def update_datatable():
    """更新结构化数据表下拉框选项"""
    return gr.update(choices=get_label_list(is_structured=True))