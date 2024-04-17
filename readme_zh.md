<a id="top"></a>
# 进行中的工作
[English](readme.md) | 中文 &nbsp;
[BlenderProc](https://github.com/DLR-RM/BlenderProc)
### 1. 进入项目目录:
```bash
cd /path/to/project
```
### 2. 安装BlenderProc:
```bash
pip install -e ./BlenderProc
```
### 3. 安装所需包:
```bash
blenderproc pip install pandas tqdm debugpy openpyxl
```
- 如果报错，尝试:
```bash
python -m blenderproc pip install pandas tqdm debugpy openpyxl
```
### 4. 运行 `dataset_generation.py`:
```bash
blenderproc run dataset_generation.py
```
- 如果报错，尝试:
```bash
python -m blenderproc run dataset_generation.py
```
[返回顶部](#top)