#novel_generator/series.py
# -*- coding: utf-8 -*-
"""
小说总体架构生成（Novel_series_generate 及相关辅助函数）
"""
import os
import json
import logging
import traceback
from novel_generator.common import invoke_with_cleaning
from llm_adapters import create_llm_adapter
from prompt_definitions import (
    series_blueprint_prompt,
    series_character_arc_prompt
)
from utils import (
    clear_file_content, 
    save_string_to_txt,
    save_data_to_json,
    load_data_from_json
    )

def Novel_series_generate(
    interface_format: str,
    api_key: str,
    base_url: str,
    llm_model: str,
    topic: str,
    genre: str,
    num_stories: int,
    filepath: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: int = 600
) -> None:
    """
    依次调用:
      1. series_blueprint_prompt
      2. character_dynamics_prompt
      3. world_building_prompt
      4. plot_series_prompt
    若在中间任何一步报错且重试多次失败，则将已经生成的内容写入 partial_series.json 并退出；
    下次调用时可从该步骤继续。
    最终输出 Novel_series.txt

    新增：
    - 在完成角色动力学设定后，依据该角色体系，使用 create_character_state_prompt 生成初始角色状态表，
      并存储到 character_state.txt，后续维护更新。
    """
    os.makedirs(filepath, exist_ok=True)
    partial_data_path = os.path.join(filepath, "partial_series.json")
    partial_data = load_data_from_json(partial_data_path)
    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=llm_model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )
    # Step1: 系列蓝图设定
    if "series_blueprint_result" not in partial_data:
        logging.info("Step1: Generating series_blueprint (系列蓝图设定) ...")
        prompt_core = series_blueprint_prompt.format(
            topic=topic,
            genre=genre,
            num_stories=num_stories,
        )
        series_blueprint_result = invoke_with_cleaning(llm_adapter, prompt_core)
        if not series_blueprint_result.strip():
            logging.warning("series_blueprint generation failed and returned empty.")
            save_data_to_json(partial_data, partial_data_path)
            return
        partial_data["series_blueprint_result"] = series_blueprint_result
        save_data_to_json(partial_data, partial_data_path)
    else:
        logging.info("Step1 already done. Skipping...")
    # Step2: 系列角色设定
    if "series_character_arc_result" not in partial_data:
        logging.info("Step2: Generating series_character_arc_prompt ...")
        prompt_character = series_character_arc_prompt.format(
            series_blueprint=partial_data["series_blueprint_result"].strip(),
            num_characters=num_stories * 2)
        series_character_arc_result = invoke_with_cleaning(llm_adapter, prompt_character)
        if not series_character_arc_result.strip():
            logging.warning("series_character_arc_prompt generation failed.")
            save_data_to_json(partial_data, partial_data_path)
            return
        partial_data["series_character_arc_result"] = series_character_arc_result
        save_data_to_json(partial_data, partial_data_path)
    else:
        logging.info("Step2 already done. Skipping...")

    series_blueprint_result = partial_data["series_blueprint_result"]
    series_character_arc_result = partial_data["series_character_arc_result"]

    final_content = (
        "#=== 0) 小说设定 ===\n"
        f"主题：{topic},类型：{genre},篇幅：约包含{num_stories}篇故事\n\n"
        "#=== 1) 系列设定 ===\n"
        f"{series_blueprint_result}\n\n"
        "#=== 2) 主要角色设定 ===\n"
        f"{series_character_arc_result}\n"
    )

    arch_file = os.path.join(filepath, "Novel_series.txt")
    clear_file_content(arch_file)
    save_string_to_txt(final_content, arch_file)
    logging.info("Novel_series.txt has been generated successfully.")

    final_json_file = os.path.join(filepath, "Novel_series.json")
    if os.path.exists(partial_data_path):
        os.rename(partial_data_path, final_json_file)
        logging.info("partial_series.json has been renamed to Novel_series.json (all steps completed).")
