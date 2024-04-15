Enter project directory:
```bash
cd /path/to/project
```
Install BlenderProc:
```bash
pip install -e ./BlenderProc
```
Install required packages:
```bash
blenderproc pip install pandas tqdm debugpy openpyxl
```
if error occurs, try:
```bash
python -m blenderproc pip install pandas tqdm debugpy openpyxl
```
Run `dataset_generation.py`:
```bash
blenderproc run dataset_generation.py
```
if error occurs, try:
```bash
python -m blenderproc run dataset_generation.py
```