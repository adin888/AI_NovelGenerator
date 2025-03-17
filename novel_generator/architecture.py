#novel_generator/architecture.pySeries_blueprint.txt
# -*- coding: utf-8 -*-
"""
小说总体架构生成（Novel_architecture_generate 及相关辅助函数）
"""
import os
import json
import logging
import traceback
from novel_generator.common import invoke_with_cleaning
from llm_adapters import create_llm_adapter
from prompt_definitions import (
    core_seed_prompt,
    character_dynamics_prompt,
    world_building_prompt,
    plot_architecture_prompt,
    create_character_state_prompt
)
from utils import (
    read_file, 
    clear_file_content, 
    save_string_to_txt,
    load_data_from_json,
    save_data_to_json
    )

def Novel_architecture_generate(
    interface_format: str,
    api_key: str,
    base_url: str,
    llm_model: str,
    story_index: int,
    number_of_chapters: int,
    word_number: int,
    filepath: str,
    user_guidance: str = "",  # 新增参数
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
    series_blueprint_file = os.path.join(filepath, "Novel_series.json")
    if not os.path.exists(series_blueprint_file):
        logging.warning("Novel_series.txt not found. Please generate series blueprint first.")
        return
    series_settings_data = load_data_from_json(series_blueprint_file)
    series_blueprint_text = series_settings_data["series_blueprint_result"].strip()
    series_character_arc_text = series_settings_data["series_character_arc_result"].strip()
    if not series_blueprint_text:
        logging.warning("Novel_series.txt is empty.")
        return
    
    partial_data_path = os.path.join(filepath, "partial_architecture.json")
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
    # Step1: 核心种子
    if "core_seed_result" not in partial_data:
        logging.info("Step1: Generating core_seed_prompt (核心种子) ...")
        prompt_core = core_seed_prompt.format(
            series_blueprint=series_blueprint_text,
            number_of_chapters=number_of_chapters,
            word_number=word_number,
            user_guidance=user_guidance,  # 修复：添加内容指导
            story_index=story_index
        )
        core_seed_result = invoke_with_cleaning(llm_adapter, prompt_core)
        if not core_seed_result.strip():
            logging.warning("core_seed_prompt generation failed and returned empty.")
            save_data_to_json(partial_data, partial_data_path)
            return
        partial_data["core_seed_result"] = core_seed_result
        save_data_to_json(partial_data, partial_data_path)
    else:
        logging.info("Step1 already done. Skipping...")
    # Step2: 角色动力学
    if "character_dynamics_result" not in partial_data:
        logging.info("Step2: Generating character_dynamics_prompt ...")
        prompt_character = character_dynamics_prompt.format(
            core_seed=partial_data["core_seed_result"].strip(),
            user_guidance=user_guidance,
            series_character_arc=series_character_arc_text)
        character_dynamics_result = invoke_with_cleaning(llm_adapter, prompt_character)
        if not character_dynamics_result.strip():
            logging.warning("character_dynamics_prompt generation failed.")
            save_data_to_json(partial_data, partial_data_path)
            return
        partial_data["character_dynamics_result"] = character_dynamics_result
        save_data_to_json(partial_data, partial_data_path)
    else:
        logging.info("Step2 already done. Skipping...")
    # 生成初始角色状态
    if "character_dynamics_result" in partial_data and "character_state_result" not in partial_data:
        logging.info("Generating initial character state from character dynamics ...")
        prompt_char_state_init = create_character_state_prompt.format(
            character_dynamics=partial_data["character_dynamics_result"].strip()
        )
        character_state_init = invoke_with_cleaning(llm_adapter, prompt_char_state_init)
        if not character_state_init.strip():
            logging.warning("create_character_state_prompt generation failed.")
            save_data_to_json(partial_data, partial_data_path)
            return
        partial_data["character_state_result"] = character_state_init
        character_state_file = os.path.join(filepath, "character_state.txt")
        clear_file_content(character_state_file)
        save_string_to_txt(character_state_init, character_state_file)
        save_data_to_json(partial_data, partial_data_path)
        logging.info("Initial character state created and saved.")
    # Step3: 世界观
    if "world_building_result" not in partial_data:
        logging.info("Step3: Generating world_building_prompt ...")
        prompt_world = world_building_prompt.format(
            core_seed=partial_data["core_seed_result"].strip(),
            user_guidance=user_guidance,  # 修复：添加用户指导
            series_blueprint=series_blueprint_text,
            core_seed=partial_data["core_seed_result"].strip())
        world_building_result = invoke_with_cleaning(llm_adapter, prompt_world)
        if not world_building_result.strip():
            logging.warning("world_building_prompt generation failed.")
            save_data_to_json(partial_data, partial_data_path)
            return
        partial_data["world_building_result"] = world_building_result
        save_data_to_json(partial_data, partial_data_path)
    else:
        logging.info("Step3 already done. Skipping...")
    # Step4: 三幕式情节
    if "plot_arch_result" not in partial_data:
        logging.info("Step4: Generating plot_architecture_prompt ...")
        prompt_plot = plot_architecture_prompt.format(
            core_seed=partial_data["core_seed_result"].strip(),
            character_dynamics=partial_data["character_dynamics_result"].strip(),
            world_building=partial_data["world_building_result"].strip(),
            user_guidance=user_guidance  # 修复：添加用户指导
        )
        plot_arch_result = invoke_with_cleaning(llm_adapter, prompt_plot)
        if not plot_arch_result.strip():
            logging.warning("plot_architecture_prompt generation failed.")
            save_data_to_json(partial_data, partial_data_path)
            return
        partial_data["plot_arch_result"] = plot_arch_result
        save_data_to_json(partial_data, partial_data_path)
    else:
        logging.info("Step4 already done. Skipping...")

    core_seed_result = partial_data["core_seed_result"]
    character_dynamics_result = partial_data["character_dynamics_result"]
    world_building_result = partial_data["world_building_result"]
    plot_arch_result = partial_data["plot_arch_result"]

    final_content = (
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

    final_json_file = os.path.join(filepath, "Novel_architecture.json")
    if os.path.exists(partial_data_path):
        os.rename(partial_data_path, final_json_file)
        logging.info("partial_architecture.json has been renamed to Novel_architecture.json (all steps completed).")
