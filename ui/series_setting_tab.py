# ui/series_setting_tab.py
# -*- coding: utf-8 -*-
import os
import json
import customtkinter as ctk
from tkinter import messagebox
from utils import (
    read_file, 
    save_string_to_txt, 
    clear_file_content, 
    extract_section,
    save_data_to_json
)
from ui.context_menu import TextWidgetContextMenu

def build_series_setting_tab(self):
    self.series_setting_tab = self.tabview.add("Series Blueprint")
    self.series_setting_tab.rowconfigure(0, weight=0)
    self.series_setting_tab.rowconfigure(1, weight=1)
    self.series_setting_tab.columnconfigure(0, weight=1)

    load_btn = ctk.CTkButton(self.series_setting_tab, text="加载 Novel_series.txt", command=self.load_series_blueprint, font=("Microsoft YaHei", 12))
    load_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")

    save_btn = ctk.CTkButton(self.series_setting_tab, text="保存修改", command=self.save_series_blueprint, font=("Microsoft YaHei", 12))
    save_btn.grid(row=0, column=0, padx=5, pady=5, sticky="e")

    self.series_setting_text = ctk.CTkTextbox(self.series_setting_tab, wrap="word", font=("Microsoft YaHei", 12))
    TextWidgetContextMenu(self.series_setting_text)
    self.series_setting_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

def load_series_blueprint(self):
    filepath = self.filepath_var.get().strip()
    if not filepath:
        messagebox.showwarning("警告", "请先设置保存文件路径")
        return
    filename = os.path.join(filepath, "Novel_series.txt")
    if not os.path.exists(filename):
        messagebox.showwarning("警告", f"未找到文件: {filename}")
        self.log(f"未找到文件: {filename}")
        return
    content = read_file(filename)
    self.series_setting_text.delete("0.0", "end")
    self.series_setting_text.insert("0.0", content)
    self.log("已加载 Novel_series.txt 内容到编辑区。")

def save_series_blueprint(self):
    filepath = self.filepath_var.get().strip()
    if not filepath:
        messagebox.showwarning("警告", "请先设置保存文件路径。")
        return
    content = self.series_setting_text.get("0.0", "end").strip()
    filename = os.path.join(filepath, "Novel_series.txt")
    clear_file_content(filename)
    save_string_to_txt(content, filename)
    self.log("已保存对 Novel_series.txt 的修改。")

    # 更新 Novel_series.json
    json_filename = os.path.join(filepath, "Novel_series.json")
    if not os.path.exists(json_filename):
        messagebox.showwarning("警告", "Novel_series.json 文件不存在。")
        self.log(f"未找到文件: {json_filename}")
        return

    try:
        with open(json_filename, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # 解析 content，更新 json_data 中的 series_blueprint_result 和 series_character_arc_result 部分
        series_blueprint_result = extract_section(content, "#=== 1) 系列设定 ===", "#=== 2) 主要角色设定 ===")
        series_character_arc_result = extract_section(content, "#=== 2) 主要角色设定 ===", None)

        if series_blueprint_result is None or series_character_arc_result is None:
            messagebox.showerror("错误", "Novel_series.txt 内容格式不正确。")
            return

        json_data["series_blueprint_result"] = series_blueprint_result
        json_data["series_character_arc_result"] = series_character_arc_result

        save_data_to_json(json_data, json_filename)
        self.log("已同步更新 Novel_series.json。")
    except Exception as e:
        messagebox.showerror("错误", f"更新 Novel_series.json 时出错: {e}")