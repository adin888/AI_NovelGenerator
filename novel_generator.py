# novel_generator.py
# -*- coding: utf-8 -*-
import os
import logging
import re
import time
import traceback
import json
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain.docstore.document import Document

# nltk、sentence_transformers 及文本处理相关
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 工具函数
from utils import (
    read_file, append_text_to_file, clear_file_content,
    save_string_to_txt
)

# prompt模板
from prompt_definitions import (
    core_seed_prompt,
    character_dynamics_prompt,
    world_building_prompt,
    plot_architecture_prompt,
    chapter_blueprint_prompt,
    chunked_chapter_blueprint_prompt,
    summary_prompt,
    update_character_state_prompt,
    first_chapter_draft_prompt,
    next_chapter_draft_prompt,
    summarize_recent_chapters_prompt,
    create_character_state_prompt
)

# 章节目录解析
from chapter_directory_parser import get_chapter_info_from_blueprint

from llm_adapters import create_llm_adapter
from embedding_adapters import create_embedding_adapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ============ 通用的重试封装 ============

def call_with_retry(func, max_retries=3, sleep_time=2, fallback_return=None, **kwargs):
    """
    通用的重试机制封装。
    :param func: 要执行的函数
    :param max_retries: 最大重试次数
    :param sleep_time: 重试前的等待秒数
    :param fallback_return: 如果多次重试仍失败时的返回值
    :param kwargs: 传给func的命名参数
    :return: func的结果，若失败则返回 fallback_return
    """
    for attempt in range(1, max_retries + 1):
        try:
            return func(**kwargs)
        except Exception as e:
            logging.warning(f"[call_with_retry] Attempt {attempt} failed with error: {e}")
            traceback.print_exc()
            if attempt < max_retries:
                time.sleep(sleep_time)
            else:
                logging.error("Max retries reached, returning fallback_return.")
                return fallback_return


# ============ 工具函数 ============

def remove_think_tags(text: str) -> str:
    """移除 <think>...</think> 包裹的内容"""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def debug_log(prompt: str, response_content: str):
    logging.info(
        f"\n[#########################################  Prompt  #########################################]\n{prompt}\n"
    )
    logging.info(
        f"\n[######################################### Response #########################################]\n{response_content}\n"
    )

def invoke_with_cleaning(llm_adapter, prompt: str) -> str:
    """
    对 LLM 的调用增加了重试封装，
    如果多次失败，则返回空字符串以继续流程，而不是中断。
    """
    def _invoke(prompt):
        return llm_adapter.invoke(prompt)

    response = call_with_retry(func=_invoke, max_retries=3, fallback_return="", prompt=prompt)
    if not response:
        logging.warning("No response from model after retry. Return empty.")
        return ""
    cleaned_text = remove_think_tags(response)
    debug_log(prompt, cleaned_text)
    return cleaned_text.strip()


# ============ 获取 vectorstore 路径 ============

def get_vectorstore_dir(filepath: str) -> str:
    return os.path.join(filepath, "vectorstore")


# ============ 清空向量库 ============

def clear_vector_store(filepath: str) -> bool:
    import shutil
    store_dir = get_vectorstore_dir(filepath)
    if not os.path.exists(store_dir):
        logging.info("No vector store found to clear.")
        return False
    try:
        shutil.rmtree(store_dir)
        logging.info(f"Vector store directory '{store_dir}' removed.")
        return True
    except Exception as e:
        logging.error(f"无法删除向量库文件夹，请关闭程序后手动删除 {store_dir}。\n {str(e)}")
        traceback.print_exc()
        return False


# ============ 根据 embedding 接口创建/加载 Chroma ============

def init_vector_store(
    embedding_adapter,
    texts: List[str],
    filepath: str
) -> Optional[Chroma]:
    """
    在 filepath 下创建/加载一个 Chroma 向量库并插入 texts。
    如果Embedding失败，则返回 None，不中断任务。
    """
    from langchain.embeddings.base import Embeddings as LCEmbeddings

    store_dir = get_vectorstore_dir(filepath)
    os.makedirs(store_dir, exist_ok=True)

    documents = [Document(page_content=str(t)) for t in texts]

    try:
        class LCEmbeddingWrapper(LCEmbeddings):
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return call_with_retry(
                    func=embedding_adapter.embed_documents,
                    max_retries=3,
                    fallback_return=[],
                    texts=texts
                )

            def embed_query(self, query: str) -> List[float]:
                res = call_with_retry(
                    func=embedding_adapter.embed_query,
                    max_retries=3,
                    fallback_return=[],
                    query=query
                )
                return res

        chroma_embedding = LCEmbeddingWrapper()

        vectorstore = Chroma.from_documents(
            documents,
            embedding=chroma_embedding,
            persist_directory=store_dir,
            client_settings=Settings(anonymized_telemetry=False),
            collection_name="novel_collection"
        )
        return vectorstore
    except Exception as e:
        logging.warning(f"Init vector store failed: {e}")
        traceback.print_exc()
        return None

def load_vector_store(
    embedding_adapter,
    filepath: str
) -> Optional[Chroma]:
    """
    读取已存在的 Chroma 向量库。若不存在则返回 None。
    如果加载失败（embedding 或IO问题），则返回 None。
    """
    store_dir = get_vectorstore_dir(filepath)
    if not os.path.exists(store_dir):
        logging.info("Vector store not found. Will return None.")
        return None

    from langchain.embeddings.base import Embeddings as LCEmbeddings

    try:
        class LCEmbeddingWrapper(LCEmbeddings):
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return call_with_retry(
                    func=embedding_adapter.embed_documents,
                    max_retries=3,
                    fallback_return=[],
                    texts=texts
                )

            def embed_query(self, query: str) -> List[float]:
                res = call_with_retry(
                    func=embedding_adapter.embed_query,
                    max_retries=3,
                    fallback_return=[],
                    query=query
                )
                return res

        chroma_embedding = LCEmbeddingWrapper()

        return Chroma(
            persist_directory=store_dir,
            embedding_function=chroma_embedding,
            client_settings=Settings(anonymized_telemetry=False),
            collection_name="novel_collection"
        )
    except Exception as e:
        logging.warning(f"Failed to load vector store: {e}")
        traceback.print_exc()
        return None


# ============ 文本分段工具 ============

def split_by_length(text: str, max_length: int = 500) -> List[str]:
    segments = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = min(start_idx + max_length, len(text))
        segment = text[start_idx:end_idx]
        segments.append(segment.strip())
        start_idx = end_idx
    return segments

def split_text_for_vectorstore(chapter_text: str,
                               max_length: int = 500,
                               similarity_threshold: float = 0.7) -> List[str]:
    """
    对新的章节文本进行分段后，再用于存入向量库。
    先句子切分 -> 语义相似度合并 -> 再按 max_length 切分。
    """
    if not chapter_text.strip():
        return []
    
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    sentences = nltk.sent_tokenize(chapter_text)
    if not sentences:
        return []

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(sentences)

    merged_paragraphs = []
    current_sentences = [sentences[0]]
    current_embedding = embeddings[0]

    for i in range(1, len(sentences)):
        sim = cosine_similarity([current_embedding], [embeddings[i]])[0][0]
        if sim >= similarity_threshold:
            current_sentences.append(sentences[i])
            current_embedding = (current_embedding + embeddings[i]) / 2.0
        else:
            merged_paragraphs.append(" ".join(current_sentences))
            current_sentences = [sentences[i]]
            current_embedding = embeddings[i]

    if current_sentences:
        merged_paragraphs.append(" ".join(current_sentences))

    final_segments = []
    for para in merged_paragraphs:
        if len(para) > max_length:
            sub_segments = split_by_length(para, max_length=max_length)
            final_segments.extend(sub_segments)
        else:
            final_segments.append(para)

    return final_segments


# ============ 更新向量库 ============

def update_vector_store(
    embedding_adapter,
    new_chapter: str,
    filepath: str
):
    """
    将最新章节文本插入到向量库中。
    若库不存在则初始化；若初始化/更新失败，则跳过。
    """
    splitted_texts = split_text_for_vectorstore(new_chapter)
    if not splitted_texts:
        logging.warning("No valid text to insert into vector store. Skipping.")
        return

    store = load_vector_store(embedding_adapter, filepath)
    if not store:
        logging.info("Vector store does not exist or failed to load. Initializing a new one for new chapter...")
        store = init_vector_store(embedding_adapter, splitted_texts, filepath)
        if not store:
            logging.warning("Init vector store failed, skip embedding.")
        else:
            logging.info("New vector store created successfully.")
        return

    # 如果已有store，则直接往里插入
    try:
        docs = [Document(page_content=str(t)) for t in splitted_texts]
        store.add_documents(docs)
        logging.info("Vector store updated with the new chapter splitted segments.")
    except Exception as e:
        logging.warning(f"Failed to update vector store: {e}")
        traceback.print_exc()


# ============ 向量检索上下文 ============

def get_relevant_context_from_vector_store(
    embedding_adapter,
    query: str,
    filepath: str,
    k: int = 2
) -> str:
    """
    从向量库中检索与 query 最相关的 k 条文本，拼接后返回。
    如果向量库加载/检索失败，则返回空字符串。
    最终只返回最多2000字符的检索片段。
    """
    store = load_vector_store(embedding_adapter, filepath)
    if not store:
        logging.info("No vector store found or load failed. Returning empty context.")
        return ""

    try:
        docs = store.similarity_search(query, k=k)
        if not docs:
            logging.info(f"No relevant documents found for query '{query}'. Returning empty context.")
            return ""
        combined = "\n".join([d.page_content for d in docs])
        # 限制长度最多2000字符
        if len(combined) > 2000:
            combined = combined[:2000]
        return combined
    except Exception as e:
        logging.warning(f"Similarity search failed: {e}")
        traceback.print_exc()
        return ""


# ============ 从目录中获取最近 n 章文本 ============

def get_last_n_chapters_text(chapters_dir: str, current_chapter_num: int, n: int = 3) -> List[str]:
    texts = []
    start_chap = max(1, current_chapter_num - n)
    for c in range(start_chap, current_chapter_num):
        chap_file = os.path.join(chapters_dir, f"chapter_{c}.txt")
        if os.path.exists(chap_file):
            text = read_file(chap_file).strip()
            texts.append(text)
        else:
            texts.append("")
    return texts


# ============ 提炼(短期摘要, 下一章关键字) ============

def summarize_recent_chapters(
    interface_format: str,
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    chapters_text_list: List[str],
    timeout: int = 600
) -> Tuple[str, str]:
    """
    生成 (short_summary, next_chapter_keywords)
    如果解析失败，则返回 (合并文本, "")
    """
    combined_text = "\n".join(chapters_text_list).strip()
    if not combined_text:
        return ("", "")

    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )

    prompt = summarize_recent_chapters_prompt.format(combined_text=combined_text)
    response_text = invoke_with_cleaning(llm_adapter, prompt)

    short_summary = ""
    next_chapter_keywords = ""

    for line in response_text.splitlines():
        line = line.strip()
        if line.startswith("短期摘要:"):
            short_summary = line.replace("短期摘要:", "").strip()
        elif line.startswith("下一章关键字:"):
            next_chapter_keywords = line.replace("下一章关键字:", "").strip()

    if not short_summary and not next_chapter_keywords:
        short_summary = response_text

    return (short_summary, next_chapter_keywords)


# ============ 持久化：情节架构（partial_architecture.json） ============

def load_partial_architecture_data(filepath: str) -> dict:
    """
    从 filepath 下的 partial_architecture.json 读取已有的阶段性数据。
    如果文件不存在或无法解析，返回空 dict。
    """
    partial_file = os.path.join(filepath, "partial_architecture.json")
    if not os.path.exists(partial_file):
        return {}

    try:
        with open(partial_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.warning(f"Failed to load partial_architecture.json: {e}")
        return {}

def save_partial_architecture_data(filepath: str, data: dict):
    """
    将阶段性数据写入 partial_architecture.json。
    """
    partial_file = os.path.join(filepath, "partial_architecture.json")
    try:
        with open(partial_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Failed to save partial_architecture.json: {e}")


# ============ 1) 生成总体架构 ============

def Novel_architecture_generate(
    interface_format: str,
    api_key: str,
    base_url: str,
    llm_model: str,
    topic: str,
    genre: str,
    number_of_chapters: int,
    word_number: int,
    filepath: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: int = 600
) -> None:
    """
    依次调用:
      1. core_seed_prompt
      2. character_dynamics_prompt
      3. world_building_prompt
      4. plot_architecture_prompt
    若在中间任何一步报错且重试多次失败，则将已经生成的内容写入 partial_architecture.json 并退出；
    下次调用时可从该步骤继续。
    最终输出 Novel_architecture.txt

    新增：
    - 在完成角色动力学设定后，依据该角色体系，使用 create_character_state_prompt 生成初始角色状态表，
      并存储到 character_state.txt，后续维护更新。
    """
    os.makedirs(filepath, exist_ok=True)

    # 加载已有的阶段性数据
    partial_data = load_partial_architecture_data(filepath)

    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=llm_model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )

    # Step1: 核心种子
    if "core_seed_result" not in partial_data:
        logging.info("Step1: Generating core_seed_prompt (核心种子) ...")
        prompt_core = core_seed_prompt.format(
            topic=topic,
            genre=genre,
            number_of_chapters=number_of_chapters,
            word_number=word_number
        )
        core_seed_result = invoke_with_cleaning(llm_adapter, prompt_core)
        if not core_seed_result.strip():
            # 多次重试依旧失败，则写入已完成内容后退出
            logging.warning("core_seed_prompt generation failed and returned empty.")
            save_partial_architecture_data(filepath, partial_data)
            return
        partial_data["core_seed_result"] = core_seed_result
        save_partial_architecture_data(filepath, partial_data)
    else:
        logging.info("Step1 already done. Skipping...")

    # Step2: 角色动力学
    # 或许可以添加不同的角色构建理论
    if "character_dynamics_result" not in partial_data:
        logging.info("Step2: Generating character_dynamics_prompt ...")
        prompt_character = character_dynamics_prompt.format(core_seed=partial_data["core_seed_result"].strip())
        character_dynamics_result = invoke_with_cleaning(llm_adapter, prompt_character)
        if not character_dynamics_result.strip():
            logging.warning("character_dynamics_prompt generation failed.")
            # 写入目前已有结果，然后退出
            save_partial_architecture_data(filepath, partial_data)
            return
        partial_data["character_dynamics_result"] = character_dynamics_result
        save_partial_architecture_data(filepath, partial_data)
    else:
        logging.info("Step2 already done. Skipping...")

    # 在完成角色动力学设定后，生成初始角色状态表
    if "character_dynamics_result" in partial_data and "character_state_result" not in partial_data:
        logging.info("Generating initial character state from character dynamics ...")
        prompt_char_state_init = create_character_state_prompt.format(
            character_dynamics=partial_data["character_dynamics_result"].strip()
        )
        character_state_init = invoke_with_cleaning(llm_adapter, prompt_char_state_init)
        if not character_state_init.strip():
            logging.warning("create_character_state_prompt generation failed.")
            # 写入目前已有结果，然后退出
            save_partial_architecture_data(filepath, partial_data)
            return

        partial_data["character_state_result"] = character_state_init
        # 保存到文件
        character_state_file = os.path.join(filepath, "character_state.txt")
        clear_file_content(character_state_file)
        save_string_to_txt(character_state_init, character_state_file)

        save_partial_architecture_data(filepath, partial_data)
        logging.info("Initial character state created and saved.")

    # Step3: 世界观
    if "world_building_result" not in partial_data:
        logging.info("Step3: Generating world_building_prompt ...")
        prompt_world = world_building_prompt.format(core_seed=partial_data["core_seed_result"].strip())
        world_building_result = invoke_with_cleaning(llm_adapter, prompt_world)
        if not world_building_result.strip():
            logging.warning("world_building_prompt generation failed.")
            save_partial_architecture_data(filepath, partial_data)
            return
        partial_data["world_building_result"] = world_building_result
        save_partial_architecture_data(filepath, partial_data)
    else:
        logging.info("Step3 already done. Skipping...")

    # Step4: 三幕式情节
    # 或许可以进一步拓展，使用不同的写法构造剧情
    if "plot_arch_result" not in partial_data:
        logging.info("Step4: Generating plot_architecture_prompt ...")
        prompt_plot = plot_architecture_prompt.format(
            core_seed=partial_data["core_seed_result"].strip(),
            character_dynamics=partial_data["character_dynamics_result"].strip(),
            world_building=partial_data["world_building_result"].strip()
        )
        plot_arch_result = invoke_with_cleaning(llm_adapter, prompt_plot)
        if not plot_arch_result.strip():
            logging.warning("plot_architecture_prompt generation failed.")
            save_partial_architecture_data(filepath, partial_data)
            return
        partial_data["plot_arch_result"] = plot_arch_result
        save_partial_architecture_data(filepath, partial_data)
    else:
        logging.info("Step4 already done. Skipping...")

    # 如果能走到这里，说明全部步骤都完成了
    core_seed_result = partial_data["core_seed_result"]
    character_dynamics_result = partial_data["character_dynamics_result"]
    world_building_result = partial_data["world_building_result"]
    plot_arch_result = partial_data["plot_arch_result"]

    final_content = (
        "#=== 0) 小说设定 ===\n"
        f"主题：{topic},类型：{genre},篇幅：约{number_of_chapters}章（每章{word_number}字）\n\n"
        "#=== 1) 核心种子 ===\n"
        f"{core_seed_result}\n\n"
        "#=== 2) 角色动力学 ===\n"
        f"{character_dynamics_result}\n\n"
        "#=== 3) 世界观 ===\n"
        f"{world_building_result}\n\n"
        "#=== 4) 三幕式情节架构 ===\n"
        f"{plot_arch_result}\n"
    )

    arch_file = os.path.join(filepath, "Novel_architecture.txt")
    clear_file_content(arch_file)
    save_string_to_txt(final_content, arch_file)
    logging.info("Novel_architecture.txt has been generated successfully.")

    # 全部生成完成后，可以考虑删除 partial_architecture.json，或保留做追溯
    # 这里选择删除
    partial_arch_file = os.path.join(filepath, "partial_architecture.json")
    if os.path.exists(partial_arch_file):
        os.remove(partial_arch_file)
        logging.info("partial_architecture.json removed (all steps completed).")


# ============ 计算分块大小的工具函数 ============

def compute_chunk_size(number_of_chapters: int, max_tokens: int) -> int:
    """
    基于“每章约100 tokens”的粗略估算，
    再结合当前max_tokens，计算分块大小：
      chunk_size = (floor(max_tokens/100/10)*10) - 10
    并确保 chunk_size 不会小于1或大于实际章节数。
    """
    tokens_per_chapter = 100.0
    ratio = max_tokens / tokens_per_chapter
    ratio_rounded_to_10 = int(ratio // 10) * 10
    chunk_size = ratio_rounded_to_10 - 10
    if chunk_size < 1:
        chunk_size = 1
    if chunk_size > number_of_chapters:
        chunk_size = number_of_chapters
    return chunk_size


def limit_chapter_blueprint(blueprint_text: str, limit_chapters: int = 100) -> str:
    """
    从已有章节目录中只取最近的 limit_chapters 章，以避免 prompt 超长。
    """
    pattern = r"(第\s*\d+\s*章.*?)(?=第\s*\d+\s*章|$)"
    chapters = re.findall(pattern, blueprint_text, flags=re.DOTALL)
    if not chapters:
        return blueprint_text

    if len(chapters) <= limit_chapters:
        return blueprint_text

    selected = chapters[-limit_chapters:]
    return "\n\n".join(selected).strip()


# ============ 2) 生成章节蓝图（新增分块逻辑 + 断点续跑） ============

def Chapter_blueprint_generate(
    interface_format: str,
    api_key: str,
    base_url: str,
    llm_model: str,
    filepath: str,
    number_of_chapters: int,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout: int = 600
) -> None:
    """
    若 Novel_directory.txt 已存在且内容非空，则表示可能是之前的部分生成结果；
      解析其中已有的章节数，从下一个章节继续分块生成；
      对于已有章节目录，传入时仅保留最近100章目录，避免prompt过长。
    否则：
      - 若章节数 <= chunk_size，直接一次性生成
      - 若章节数 > chunk_size，进行分块生成
    生成完成后输出至 Novel_directory.txt。
    """
    arch_file = os.path.join(filepath, "Novel_architecture.txt")
    if not os.path.exists(arch_file):
        logging.warning("Novel_architecture.txt not found. Please generate architecture first.")
        return

    architecture_text = read_file(arch_file).strip()
    if not architecture_text:
        logging.warning("Novel_architecture.txt is empty.")
        return

    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=llm_model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )

    filename_dir = os.path.join(filepath, "Novel_directory.txt")
    if not os.path.exists(filename_dir):
        # 如果文件不存在，就先建一个空文件
        open(filename_dir, "w", encoding="utf-8").close()

    existing_blueprint = read_file(filename_dir).strip()
    chunk_size = compute_chunk_size(number_of_chapters, max_tokens)
    logging.info(f"Number of chapters = {number_of_chapters}, computed chunk_size = {chunk_size}.")

    # 如果已经有部分章节蓝图生成了，则进行断点续跑
    if existing_blueprint:
        logging.info("Detected existing blueprint content. Will resume chunked generation from that point.")

        pattern = r"第\s*(\d+)\s*章"
        existing_chapter_numbers = re.findall(pattern, existing_blueprint)
        existing_chapter_numbers = [int(x) for x in existing_chapter_numbers if x.isdigit()]

        if existing_chapter_numbers:
            max_existing_chap = max(existing_chapter_numbers)
        else:
            max_existing_chap = 0

        logging.info(f"Existing blueprint indicates up to chapter {max_existing_chap} has been generated.")

        final_blueprint = existing_blueprint
        current_start = max_existing_chap + 1
        while current_start <= number_of_chapters:
            current_end = min(current_start + chunk_size - 1, number_of_chapters)
            limited_blueprint = limit_chapter_blueprint(final_blueprint, 100)

            chunk_prompt = chunked_chapter_blueprint_prompt.format(
                novel_architecture=architecture_text,
                chapter_list=limited_blueprint,  # 只保留最近100章
                number_of_chapters=number_of_chapters,
                n=current_start,
                m=current_end
            )
            logging.info(f"Generating chapters [{current_start}..{current_end}] in a chunk...")

            chunk_result = invoke_with_cleaning(llm_adapter, chunk_prompt)
            if not chunk_result.strip():
                logging.warning(f"Chunk generation for chapters [{current_start}..{current_end}] is empty.")
                # 写入当前已经有的 final_blueprint，并结束
                clear_file_content(filename_dir)
                save_string_to_txt(final_blueprint.strip(), filename_dir)
                return

            final_blueprint += "\n\n" + chunk_result.strip()

            # 实时写入
            clear_file_content(filename_dir)
            save_string_to_txt(final_blueprint.strip(), filename_dir)

            current_start = current_end + 1

        logging.info("All chapters blueprint have been generated (resumed chunked).")
        return

    # 如果 Novel_directory.txt 为空，则分情况：
    # 1) 如果 chunk_size >= number_of_chapters，可以一次性生成
    if chunk_size >= number_of_chapters:
        prompt = chapter_blueprint_prompt.format(
            novel_architecture=architecture_text,
            number_of_chapters=number_of_chapters
        )
        blueprint_text = invoke_with_cleaning(llm_adapter, prompt)
        if not blueprint_text.strip():
            logging.warning("Chapter blueprint generation result is empty.")
            return

        clear_file_content(filename_dir)
        save_string_to_txt(blueprint_text, filename_dir)
        logging.info("Novel_directory.txt (chapter blueprint) has been generated successfully (single-shot).")
        return

    # 2) 如果 chunk_size < number_of_chapters，则进行分块生成
    logging.info("Will generate chapter blueprint in chunked mode from scratch.")
    final_blueprint = ""
    current_start = 1
    while current_start <= number_of_chapters:
        current_end = min(current_start + chunk_size - 1, number_of_chapters)
        limited_blueprint = limit_chapter_blueprint(final_blueprint, 100)

        chunk_prompt = chunked_chapter_blueprint_prompt.format(
            novel_architecture=architecture_text,
            chapter_list=limited_blueprint,  # 只保留最近100章
            number_of_chapters=number_of_chapters,
            n=current_start,
            m=current_end
        )
        logging.info(f"Generating chapters [{current_start}..{current_end}] in a chunk...")

        chunk_result = invoke_with_cleaning(llm_adapter, chunk_prompt)
        if not chunk_result.strip():
            logging.warning(f"Chunk generation for chapters [{current_start}..{current_end}] is empty.")
            # 写入已经生成的 final_blueprint
            clear_file_content(filename_dir)
            save_string_to_txt(final_blueprint.strip(), filename_dir)
            return

        if final_blueprint.strip():
            final_blueprint += "\n\n" + chunk_result.strip()
        else:
            final_blueprint = chunk_result.strip()

        # 实时写入，以免中途崩溃造成丢失
        clear_file_content(filename_dir)
        save_string_to_txt(final_blueprint.strip(), filename_dir)

        current_start = current_end + 1

    logging.info("Novel_directory.txt (chapter blueprint) has been generated successfully (chunked).")


# ============ 3) 生成章节草稿 ============

def generate_chapter_draft(
    api_key: str,
    base_url: str,
    model_name: str,
    filepath: str,
    novel_number: int,
    word_number: int,
    temperature: float,
    user_guidance: str,
    characters_involved: str,
    key_items: str,
    scene_location: str,
    time_constraint: str,
    embedding_api_key: str,
    embedding_url: str,
    embedding_interface_format: str,
    embedding_model_name: str,
    embedding_retrieval_k: int = 2,
    interface_format: str = "openai",
    max_tokens: int = 2048,
    timeout: int = 600
) -> str:
    """
    根据 novel_number 判断是否为第一章。
    - 若是第一章，则使用 first_chapter_draft_prompt
    - 否则使用 next_chapter_draft_prompt
    最终将生成文本存入 chapters/chapter_{novel_number}.txt。
    """
    arch_file = os.path.join(filepath, "Novel_architecture.txt")
    novel_architecture_text = read_file(arch_file)

    directory_file = os.path.join(filepath, "Novel_directory.txt")
    blueprint_text = read_file(directory_file)

    global_summary_file = os.path.join(filepath, "global_summary.txt")
    global_summary_text = read_file(global_summary_file)

    character_state_file = os.path.join(filepath, "character_state.txt")
    character_state_text = read_file(character_state_file)

    # 获取本章在目录中的信息
    chapter_info = get_chapter_info_from_blueprint(blueprint_text, novel_number)
    chapter_title = chapter_info["chapter_title"]
    chapter_role = chapter_info["chapter_role"]
    chapter_purpose = chapter_info["chapter_purpose"]
    suspense_level = chapter_info["suspense_level"]
    foreshadowing = chapter_info["foreshadowing"]
    plot_twist_level = chapter_info["plot_twist_level"]
    chapter_summary = chapter_info["chapter_summary"]

    # 准备章节目录文件夹
    chapters_dir = os.path.join(filepath, "chapters")
    os.makedirs(chapters_dir, exist_ok=True)

    # 判断是否为第一章
    if novel_number == 1:
        prompt_text = first_chapter_draft_prompt.format(
            novel_number=novel_number,
            word_number=word_number,
            chapter_title=chapter_title,
            chapter_role=chapter_role,
            chapter_purpose=chapter_purpose,
            suspense_level=suspense_level,
            foreshadowing=foreshadowing,
            plot_twist_level=plot_twist_level,
            chapter_summary=chapter_summary,

            characters_involved=characters_involved,
            key_items=key_items,
            scene_location=scene_location,
            time_constraint=time_constraint,
            user_guidance=user_guidance,

            novel_setting=novel_architecture_text
        )
    else:
        # 若不是第一章，则获取最近几章文本，并做摘要与检索
        recent_3_texts = get_last_n_chapters_text(chapters_dir, novel_number, n=3)
        short_summary, next_chapter_keywords = summarize_recent_chapters(
            interface_format=interface_format,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            chapters_text_list=recent_3_texts,
            timeout=timeout
        )

        # 从最近章节中获取最后一段作为前章结尾
        previous_chapter_excerpt = ""
        for text_block in reversed(recent_3_texts):
            if text_block.strip():
                # 取后1500字符左右
                if len(text_block) > 1500:
                    previous_chapter_excerpt = text_block[-1500:]
                else:
                    previous_chapter_excerpt = text_block
                break

        # 从向量库检索上下文
        embedding_adapter = create_embedding_adapter(
            embedding_interface_format,
            embedding_api_key,
            embedding_url,
            embedding_model_name
        )
        retrieval_query = short_summary + " " + next_chapter_keywords
        relevant_context = get_relevant_context_from_vector_store(
            embedding_adapter=embedding_adapter,
            query=retrieval_query,
            filepath=filepath,
            k=embedding_retrieval_k
        )
        if not relevant_context.strip():
            relevant_context = "（无检索到的上下文）"

        prompt_text = next_chapter_draft_prompt.format(
            novel_number=novel_number,
            word_number=word_number,
            chapter_title=chapter_title,
            chapter_role=chapter_role,
            chapter_purpose=chapter_purpose,
            suspense_level=suspense_level,
            foreshadowing=foreshadowing,
            plot_twist_level=plot_twist_level,
            chapter_summary=chapter_summary,

            characters_involved=characters_involved,
            key_items=key_items,
            scene_location=scene_location,
            time_constraint=time_constraint,
            user_guidance=user_guidance,

            novel_setting=novel_architecture_text,
            global_summary=global_summary_text,
            character_state=character_state_text,
            context_excerpt=relevant_context,
            previous_chapter_excerpt=previous_chapter_excerpt
        )

    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )

    chapter_content = invoke_with_cleaning(llm_adapter, prompt_text)
    if not chapter_content.strip():
        logging.warning("Generated chapter draft is empty.")

    chapter_file = os.path.join(chapters_dir, f"chapter_{novel_number}.txt")
    clear_file_content(chapter_file)
    save_string_to_txt(chapter_content, chapter_file)

    logging.info(f"[Draft] Chapter {novel_number} generated as a draft.")
    return chapter_content


# ============ 4) 定稿章节 ============

def finalize_chapter(
    novel_number: int,
    word_number: int,
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float,
    filepath: str,
    embedding_api_key: str,
    embedding_url: str,
    embedding_interface_format: str,
    embedding_model_name: str,
    interface_format: str,
    max_tokens: int,
    timeout: int = 600
):
    """
    对指定章节做最终处理：更新全局摘要、更新角色状态、插入向量库等。
    默认无需再做扩写操作，若有需要可在外部调用 enrich_chapter_text 处理后再定稿。
    """
    chapters_dir = os.path.join(filepath, "chapters")
    chapter_file = os.path.join(chapters_dir, f"chapter_{novel_number}.txt")
    chapter_text = read_file(chapter_file).strip()
    if not chapter_text:
        logging.warning(f"Chapter {novel_number} is empty, cannot finalize.")
        return

    # 进行摘要、角色状态更新
    global_summary_file = os.path.join(filepath, "global_summary.txt")
    old_global_summary = read_file(global_summary_file)

    character_state_file = os.path.join(filepath, "character_state.txt")
    old_character_state = read_file(character_state_file)

    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )

    # 更新全局摘要
    prompt_summary = summary_prompt.format(
        chapter_text=chapter_text,
        global_summary=old_global_summary
    )
    new_global_summary = invoke_with_cleaning(llm_adapter, prompt_summary)
    if not new_global_summary.strip():
        new_global_summary = old_global_summary

    # 更新角色状态
    prompt_char_state = update_character_state_prompt.format(
        chapter_text=chapter_text,
        old_state=old_character_state
    )
    new_char_state = invoke_with_cleaning(llm_adapter, prompt_char_state)
    if not new_char_state.strip():
        new_char_state = old_character_state

    clear_file_content(global_summary_file)
    save_string_to_txt(new_global_summary, global_summary_file)

    clear_file_content(character_state_file)
    save_string_to_txt(new_char_state, character_state_file)

    # 更新向量库
    embedding_adapter = create_embedding_adapter(
        embedding_interface_format,
        embedding_api_key,
        embedding_url,
        embedding_model_name
    )
    update_vector_store(embedding_adapter, chapter_text, filepath)

    logging.info(f"Chapter {novel_number} has been finalized.")


def enrich_chapter_text(
    chapter_text: str,
    word_number: int,
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float,
    interface_format: str,
    max_tokens: int,
    timeout: int=600
) -> str:
    """
    对章节文本进行扩写，使其更接近 word_number 字数，保持剧情连贯。
    """
    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )
    prompt = f"""以下章节文本较短，请在保持剧情连贯的前提下进行扩写，使其更充实，接近 {word_number} 字左右：
原内容：
{chapter_text}
"""
    enriched_text = invoke_with_cleaning(llm_adapter, prompt)
    return enriched_text if enriched_text else chapter_text


# ============ 导入知识文件到向量库 ============

def advanced_split_content(content: str,
                           similarity_threshold: float = 0.7,
                           max_length: int = 500) -> List[str]:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    sentences = nltk.sent_tokenize(content)
    if not sentences:
        return []

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(sentences)

    merged_paragraphs = []
    current_sentences = [sentences[0]]
    current_embedding = embeddings[0]

    for i in range(1, len(sentences)):
        sim = cosine_similarity([current_embedding], [embeddings[i]])[0][0]
        if sim >= similarity_threshold:
            current_sentences.append(sentences[i])
            current_embedding = (current_embedding + embeddings[i]) / 2.0
        else:
            merged_paragraphs.append(" ".join(current_sentences))
            current_sentences = [sentences[i]]
            current_embedding = embeddings[i]

    if current_sentences:
        merged_paragraphs.append(" ".join(current_sentences))

    final_segments = []
    for para in merged_paragraphs:
        if len(para) > max_length:
            sub_segments = split_by_length(para, max_length=max_length)
            final_segments.extend(sub_segments)
        else:
            final_segments.append(para)

    return final_segments

def import_knowledge_file(
    embedding_api_key: str,
    embedding_url: str,
    embedding_interface_format: str,
    embedding_model_name: str,
    file_path: str,
    filepath: str
):
    logging.info(f"开始导入知识库文件: {file_path}, 接口格式: {embedding_interface_format}, 模型: {embedding_model_name}")
    if not os.path.exists(file_path):
        logging.warning(f"知识库文件不存在: {file_path}")
        return

    content = read_file(file_path)
    if not content.strip():
        logging.warning("知识库文件内容为空。")
        return

    paragraphs = advanced_split_content(content)

    embedding_adapter = create_embedding_adapter(
        interface_format=embedding_interface_format,
        api_key=embedding_api_key,
        base_url=embedding_url if embedding_url else "http://localhost:11434/api",
        model_name=embedding_model_name
    )

    store = load_vector_store(embedding_adapter, filepath)
    if not store:
        logging.info("Vector store does not exist or load failed. Initializing a new one for knowledge import...")
        store = init_vector_store(embedding_adapter, paragraphs, filepath)
        if store:
            logging.info("知识库文件已成功导入至向量库(新初始化)。")
        else:
            logging.warning("知识库导入失败，跳过。")
    else:
        try:
            docs = [Document(page_content=str(p)) for p in paragraphs]
            store.add_documents(docs)
            logging.info("知识库文件已成功导入至向量库(追加模式)。")
        except Exception as e:
            logging.warning(f"知识库导入失败: {e}")
            traceback.print_exc()
