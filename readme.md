<a id="top"></a>
# Work in progress

A referring expression comprehension / segmentation dataset generation engine,  capable of generating synthetic photos and expressions for language-conditioned grasping. Based on Blender and [BlenderProc](https://github.com/DLR-RM/BlenderProc).  

The RefGrasp Dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/1eHsqY1xMYZPKOF-jkt1TOHodXPmYYjG8/view?usp=drive_link) or [Quark Drive](https://pan.quark.cn/s/4dea47501acf). Instructions on how to use the dataset can be found [**here**](./dataset_usage.md).

Examples:  
![fig1](/images/fig1.jpg "fig1")
![fig2](/images/fig2.jpg "fig2")
![fig3](/images/fig3.png "fig3")

# The content below is for dataset generation.
a conda environment is recommended for the project. Below is a guide to set up the environment for project and run the dataset generation process. The project has been tested on **Windows 10**, **Windows 11** and **Ubuntu 18.04**.
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
RefGrasp  
├─ <b>blender_files</b>
│  └─ <b>background.blend</b>
├─ BlenderProc  
├─ core  
├─ images 
├─ <b>models</b> 
│  └─ <b>...(290 files)</b>
├─ tool_scripts
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
`--persistent-data-cleanup-interval`: Clean up persistent data interval. Default to 100.  
`--texture-limit`: Texture limit. Default to "2048".  
`--cpu-threads`: Number of CPU threads used in render. 0 for auto setting. Default to 0.  
`--samples`: Max samples for rendering, bigger for less noise. Default to 512.

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
`--persistent-data-cleanup-interval`: Clean up persistent data interval. Default to 100.  
`--texture-limit`: Texture limit. Default to "2048".  
`--cpu-threads`: Number of CPU threads used in render for each subprocess. 0 for auto setting. Default to 16. 
`--samples`: Max samples for rendering, bigger for less noise. Default to 512.

# Acknowledgments
This project would not have been possible without the valuable work of several other open-source projects. We would like to extend our gratitude to the following repositories for their contributions, which were instrumental in the development of this project:

* [BlenderProc](https://github.com/DLR-RM/BlenderProc): Thank you for providing a procedrual rendering pipeline and other useful functionalities that was adapted and integrated into this project.
* [polygon-transformer](https://github.com/amazon-science/polygon-transformer): We utilized and modified the polygon processing module, which greatly helped in annotating the dataset.

We highly appreciate the efforts of these projects and their maintainers. Their contributions to the open-source community are invaluable.

[to top](#top)