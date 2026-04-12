# 批量推理 WebUI

为 `batch_inference.py` 提供的 Gradio 图形界面。用它可以配置批量推理、在试运行后用耳朵挑选参考音频、并一键启动正式合成——不需要手动编辑 `.list` 文件或 `.bat` 文件。

## 启动

双击仓库根目录下的 `go-batch-webui.bat`，或者执行：

```
runtime\python.exe -I webui_batch_inference.py zh_CN   :: 中文界面
runtime\python.exe -I webui_batch_inference.py en_US   :: 英文界面
```

默认监听 `9870` 端口，会自动打开浏览器。

## 标签页

### 1. 配置（Config）

| 字段 | 说明 |
|---|---|
| **输入列表通配符** | 匹配 `.list` 文件的通配符，会原样传给 `--input_list`。默认 `inputs/mod_input/*.list`。点击 **预览匹配的文件** 可以先看看到底会匹配到哪些文件。 |
| **说话人配置** | 下拉框内容来自 `inputs/*.yaml`。点击 **查看说话人列表**，会把 YAML 内容以表格形式（speaker → GPT / SoVITS / ref list 路径）展开，不用再去开文本编辑器。 |
| **输出目录** | 正式合成的 `.wav` 文件和试运行生成的 `.list` 文件都会写到这里。 |
| **候选参考音频数量 (Top-K)** | 试运行时每行保留的候选数量。默认 5。 |
| **模式** | `auto` = 多说话人、基于情感自动挑选参考音频（需要 `--speaker_config`）；`manual` = 所有行使用同一条参考音频。切换到 `manual` 后会显示参考音频 / 参考文本 / 参考语言的输入框。 |
| **高级推理参数** | `top_k`、`top_p`、`temperature`、`speed_factor`、`batch_size`、`seed`、`text_split_method`、`output_sr`、`output_channels`。默认值与 `batch_inference.py` 保持一致。 |

表单下面有两个按钮：

- **试运行（挑选参考音频）** — 以子进程方式运行 `batch_inference.py --dry_run --dry_run_topk N ...`，把 stdout/stderr 实时写到下方的 **运行日志** 里。不会合成任何音频，仅把 Top-K 候选的路径写进每个输出 `.list` 的第 5 列。
- **开始批量推理** — 同样的参数，但不加 `--dry_run`，真正把 `.wav` 写到输出目录。

### 2. 审阅试运行结果（Review dry-run output）

试运行结束后，切到这个页面试听并挑选候选参考音频。

1. 点击 **重新扫描 output/**，从下拉框里选一个 `.list` 文件。界面会自动隐藏 `.curated.list` 文件，避免把你自己保存过的结果再审阅一遍。
2. 界面一次显示 5 行。每行展示：
   - 目标文本、说话人、语言；
   - 最多 Top-K 个内嵌音频播放器（点击即可在浏览器里试听）；
   - 一个单选按钮，用于标记最合适的候选。
3. 用 **◀ 上一页** / **下一页 ▶** 浏览所有行，翻页时你已做的选择会被记住。
4. 点击 **保存筛选后的列表**。会在原文件旁边生成 `<name>.curated.list`，第 5 列只保留你选中的那一条参考音频路径。
5. 点击 **使用筛选后的列表开始批量推理**，立即用刚保存的 curated 文件做正式合成。它会再起一个子进程：`batch_inference.py --input_list <curated>`（不带 `--dry_run`）。

### 3. 帮助（Help）

上述流程的简要说明。

## 典型工作流

```
编辑 inputs/mod_input/*.list   （写入你要合成的文本）
  ↓
go-batch-webui.bat → 配置标签页 → 试运行
  ↓
审阅标签页 → 试听候选 → 每行选一条 → 保存筛选后的列表
  ↓
使用筛选后的列表开始批量推理
  ↓
output/**/*.wav
```

## 常见问题

- **日志框里出现 Unicode / GBK 相关报错** —— 启动脚本里已经设置了 `chcp 65001` 和 `PYTHONIOENCODING=utf-8`。如果你是在其他 shell 里直接跑 Python，请先自己设置好。
- **英文 G2P 提示缺少 NLTK tagger** —— 运行 `python -m nltk.downloader averaged_perceptron_tagger_eng`。
- **首次运行时情感模型下载失败** —— 请确认可以访问 Hugging Face，或者预先把模型放到 `GPT_SoVITS/pretrained_models/emotion-*`。
- **说话人配置下拉框是空的** —— 扫描器只看 `inputs/*.yaml` / `inputs/*.yml`。把你的 YAML 放到那里，或者直接在下拉框里手动输入路径（下拉框是可编辑的）。
- **某一行没有候选** —— 试运行只会为匹配成功的行写入数据。检查输入 `.list` 第 2 列的 speaker 名和 `speaker_config.yaml` 里的 key 是否完全一致（小写、大小写敏感）。
- **端口已被占用** —— 修改 `webui_batch_inference.py` 里的 `DEFAULT_PORT`，或者腾出 `9870` 端口。

## 相关文件

- `batch_inference.py` —— WebUI 所封装的底层命令行入口。运行 `python batch_inference.py --help` 可查看完整参数。
