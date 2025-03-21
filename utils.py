# utils.py
# -*- coding: utf-8 -*-
import os
import json

def read_file(filename: str) -> str:
    """读取文件的全部内容，若文件不存在或异常则返回空字符串。"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return ""
    except Exception as e:
        print(f"[read_file] 读取文件时发生错误: {e}")
        return ""

def append_text_to_file(text_to_append: str, file_path: str):
    """在文件末尾追加文本(带换行)。若文本非空且无换行，则自动加换行。"""
    if text_to_append and not text_to_append.startswith('\n'):
        text_to_append = '\n' + text_to_append

    try:
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(text_to_append)
    except IOError as e:
        print(f"[append_text_to_file] 发生错误：{e}")

def clear_file_content(filename: str):
    """清空指定文件内容。"""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            pass
    except IOError as e:
        print(f"[clear_file_content] 无法清空文件 '{filename}' 的内容：{e}")

def save_string_to_txt(content: str, filename: str):
    """将字符串保存为 txt 文件（覆盖写）。"""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
    except Exception as e:
        print(f"[save_string_to_txt] 保存文件时发生错误: {e}")

def save_data_to_json(data: dict, file_path: str) -> bool:
    """将数据保存到 JSON 文件。"""
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"保存数据到JSON文件 {file_path} 时出错: {e}")
        return False

def load_data_from_json(file_path: str) -> dict:
    """从 JSON 文件中加载数据。"""
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        print(f"从JSON文件 {file_path} 加载数据时出错: {e}")
        return {}
    
def extract_section(content, start_marker, end_marker):
    start_index = content.find(start_marker)
    if start_index == -1:
        return None
    start_index += len(start_marker)
    if end_marker:
        end_index = content.find(end_marker, start_index)
        if end_index == -1:
            return None
        return content[start_index:end_index].strip()
    else:
        return content[start_index:].strip()