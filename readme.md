<a id="top"></a>
# Work in progress
English | [中文](readme_zh.md) &nbsp;
[BlenderProc](https://github.com/DLR-RM/BlenderProc)

A referring expression comprehension / segmentation dataset generation engine,  capable of generating synthetic photos and expressions for language-conditioned grasping. Based on Blender and BlenderProc.

examples:  
![fig1](/images/fig1.jpg "fig1")
![fig2](/images/fig2.jpg "fig2")
![fig3](/images/fig3.png "fig3")

a conda environment is recommended for the project. Below is a guide to set up the environment for project and run the dataset generation process. The project has been tested on **Windows 10** and **Ubuntu 18.04**.
# Quick Start
### 1. Clone the project:
```bash
cd /path/to/project
mkdir RefGrasp && cd RefGrasp
git clone https://github.com/Soappyooo/Refer_Grasp.git .
```
### 2. Create a conda environment:
```bash
conda create -n refgrasp python==3.10.13
conda activate refgrasp
pip install -r requirements.txt
```
### 3. Install BlenderProc:
```bash
pip install -e ./BlenderProc
```
### 4. Install required packages:
It may take a while to install the packages as a Blender 3.5 would be installed first.
```bash
python -m blenderproc pip install pandas tqdm debugpy openpyxl psutil
```
### 5. Download models and scene:
Download (760MB) from [Google Drive](https://drive.google.com/file/d/1tDshqp_SNw9SoH4jtoeFkZu8dhrZSY12/view?usp=sharing) or [Quark Drive](https://pan.quark.cn/s/d94083a37db0). Extract the downloaded file to the project directory. The directory structure should look like this:
<pre>
/RefGrasp  
├─ <b>blender_files</b>
│  └─ <b>background.blend</b>
├─ BlenderProc  
├─ core  
├─ images 
├─ <b>models</b> 
│  └─ <b>...(290 files)</b>
├─ utils
└─ ...(other files)  
</pre>

### 6. Run `dataset_generation.py`:
Running the script will generate a demo dataset (about 50 images) in a `output_demo` directory.
```bash
python -m blenderproc run dataset_generation.py
```
<br>

# Dataset Generation
### Run with arguments on single GPU:
For example, to generate a dataset with 1000 iterations and 5 images per iteration, run:
```bash
python -m blenderproc run dataset_generation.py --output-dir ./output --iterations 1000 --images-per-iteration 5
```
#### All possible arguments are listed below:  
`--gpu-id`: GPU device id used for rendering. Default to 0.  
`--output-dir`: Output directory of the generated dataset. Default to "./output_demo".  
`--obj-dir`: Model files directory. Default to "./models".  
`--blender-scene-file-path`: Background scene file path. Default to "./blender_files/background.blend".  
`--models-info-file-path`: Models info file path. Default to "./models/models_info_all.xlsx".  
`--log-file-path`: Log file path. Default to "./output_demo/dataset_generation.log".  
`--iterations`: Number of iterations. Default to 10.  
`--images-per-iteration`: Number of images per iteration. Total images <= iterations * images_per_iteration. Images within an iteration shares the same scene with different background, light and camera pose. Default to 5.  
`--resolution-width`: Render resolution width. Default to 512.  
`--resolution-height`: Render resolution height. Default to 512.  
`--scene-graph-rows`: Scene graph rows. Default to 4.  
`--scene-graph-cols`: Scene graph cols. Default to 4.  
`--seed`: Random seed, should not be set unless debugging. Default to None.  
`--persistent-data-cleanup-interval`: Clean up persistent data interval. Default to 7.  
`--texture-limit`: Texture limit. Default to "2048".  
`--cpu-threads`: Number of CPU threads used in render. 0 for auto setting. Default to 0.  

### Run with arguments on multiple GPUs:
For example, to generate a dataset with 1000 iterations and 5 images per iteration on 4 GPUs, run:
```bash
python dataset_generation_multi_gpu.py --gpu-ids 0,1,2,3 --output-dir ./output --iteration 1000 --images-per-iteration 5
```
#### All possible arguments are listed below: 
`--gpu-ids`: GPU device ids used for rendering. Default to "0,1,2,3".  
`--output-dir`: Output directory of the generated dataset. Default to "./output".  
`--obj-dir`: Model files directory. Default to "./models".  
`--blender-scene-file-path`: Background scene file path. Default to "./blender_files/background.blend".  
`--models-info-file-path`: Models info file path. Default to "./models/models_info_all.xlsx".   
`--iterations`: Number of iterations per gpu. Default to 10.  
`--images-per-iteration`: Number of images per iteration. Total images <= iterations * images_per_iteration * gpu_count. Default to 5.  
`--resolution-width`: Render resolution width. Default to 512.  
`--resolution-height`: Render resolution height. Default to 512.  
`--scene-graph-rows`: Scene graph rows. Default to 4.  
`--scene-graph-cols`: Scene graph cols. Default to 4.   
`--persistent-data-cleanup-interval`: Clean up persistent data interval. Default to 7.  
`--texture-limit`: Texture limit. Default to "2048".  
`--cpu-threads`: Number of CPU threads used in render for each subprocess. 0 for auto setting. Default to 16. 


[to top](#top)