# ui/setting_tab.py
# -*- coding: utf-8 -*-
import os
import customtkinter as ctk
import json
from tkinter import messagebox
from utils import (
    read_file, 
    save_string_to_txt, 
    clear_file_content, 
    extract_section,
    save_data_to_json
)
from ui.context_menu import TextWidgetContextMenu

def build_setting_tab(self):
    self.setting_tab = self.tabview.add("Novel Architecture")
    self.setting_tab.rowconfigure(0, weight=0)
    self.setting_tab.rowconfigure(1, weight=1)
    self.setting_tab.columnconfigure(0, weight=1)

    load_btn = ctk.CTkButton(self.setting_tab, text="加载 Novel_architecture.txt", command=self.load_novel_architecture, font=("Microsoft YaHei", 12))
    load_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")

    save_btn = ctk.CTkButton(self.setting_tab, text="保存修改", command=self.save_novel_architecture, font=("Microsoft YaHei", 12))
    save_btn.grid(row=0, column=0, padx=5, pady=5, sticky="e")

    self.setting_text = ctk.CTkTextbox(self.setting_tab, wrap="word", font=("Microsoft YaHei", 12))
    TextWidgetContextMenu(self.setting_text)
    self.setting_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

def load_novel_architecture(self):
    filepath = self.filepath_var.get().strip()
    if not filepath:
        messagebox.showwarning("警告", "请先设置保存文件路径")
        return
    filename = os.path.join(filepath, "Novel_architecture.txt")
    if not os.path.exists(filename):
        messagebox.showwarning("警告", f"未找到文件: {filename}")
        self.log(f"未找到文件: {filename}")
        return
    content = read_file(filename)
    self.setting_text.delete("0.0", "end")
    self.setting_text.insert("0.0", content)
    self.log("已加载 Novel_architecture.txt 内容到编辑区。")

def save_novel_architecture(self):
    filepath = self.filepath_var.get().strip()
    if not filepath:
        messagebox.showwarning("警告", "请先设置保存文件路径。")
        return
    content = self.setting_text.get("0.0", "end").strip()
    filename = os.path.join(filepath, "Novel_architecture.txt")
    clear_file_content(filename)
    save_string_to_txt(content, filename)
    self.log("已保存对 Novel_architecture.txt 的修改。")

    # 更新 Novel_architecture.json
    json_filename = os.path.join(filepath, "Novel_architecture.json")
    if not os.path.exists(json_filename):
        messagebox.showwarning("警告", f"未找到文件: {json_filename}")
        self.log(f"未找到文件: {json_filename}")
        return
    try:
        with open(json_filename, "r", encoding="utf-8") as f:
            json_data = json.load(f)
         # 解析 content，更新 json_data 中的各个部分
        core_seed = extract_section(content, "#=== 1) 核心种子 ===", "#=== 2) 角色动力学 ===")
        character_dynamics = extract_section(content, "#=== 2) 角色动力学 ===", "#=== 3) 世界观 ===")
        world_building = extract_section(content, "#=== 3) 世界观 ===", "#=== 4) 三幕式情节架构 ===")
        plot_arch = extract_section(content, "#=== 4) 三幕式情节架构 ===", "")
        if core_seed is None or character_dynamics is None or world_building is None or plot_arch is None:
                 messagebox.showerror("错误", "Novel_architecture.txt 内容格式不正确。")
                 return
        json_data["core_seed_result"] = core_seed
        json_data["character_dynamics_result"] = character_dynamics
        json_data["world_building_result"] = world_building
        json_data["plot_architecture_result"] = plot_arch
        save_data_to_json(json_data, json_filename)
        self.log("已同步更新 Novel_architecture.json。")
    except Exception as e:
        messagebox.showerror("错误", f"更新 Novel_architecture.json 时出错: {e}")
