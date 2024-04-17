<a id="top"></a>
# Work in progress
English | [中文](readme_zh.md) &nbsp;
[BlenderProc](https://github.com/DLR-RM/BlenderProc)
### 1. Enter project directory:
```bash
cd /path/to/project
```
### 2. Install BlenderProc:
```bash
pip install -e ./BlenderProc
```
### 3. Install required packages:
```bash
blenderproc pip install pandas tqdm debugpy openpyxl
```
- if error occurs, try:
```bash
python -m blenderproc pip install pandas tqdm debugpy openpyxl
```
### 4. Run `dataset_generation.py`:
```bash
blenderproc run dataset_generation.py
```
- if error occurs, try:
```bash
python -m blenderproc run dataset_generation.py
```
[to top](#top)