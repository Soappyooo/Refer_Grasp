<a id="top"></a>
# 进行中的工作
[English](readme.md) | 中文 &nbsp;
[BlenderProc](https://github.com/DLR-RM/BlenderProc)

一个指代表达分割数据集生成引擎，能够为语言条件抓取生成合成照片和表达。基于Blender和BlenderProc。

示例:  
![fig1](/images/fig1.jpg "fig1")
![fig2](/images/fig2.jpg "fig2")
![fig3](/images/fig3.png "fig3")

建议为项目使用conda环境。以下是设置项目环境并运行数据集生成过程的指南。

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