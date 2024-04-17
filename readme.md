<a id="top"></a>
# Work in progress
English | [中文](readme_zh.md) &nbsp;
[BlenderProc](https://github.com/DLR-RM/BlenderProc)

A referring expression comprehension / segmentation dataset generation engine,  capable of generating synthetic photos and expressions for language-conditioned grasping. Based on Blender and BlenderProc.

examples:  
![fig1](/images/fig1.jpg "fig1")
![fig2](/images/fig2.jpg "fig2")
![fig3](/images/fig3.png "fig3")

a conda environment is recommended for the project. Below is a guide to set up the environment for project and run the dataset generation process.

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