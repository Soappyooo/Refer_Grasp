<a id="top"></a>
# 进行中的工作
[English](readme.md) | 中文 &nbsp;
[BlenderProc](https://github.com/DLR-RM/BlenderProc)

一个指代表达分割数据集生成引擎，能够为语言条件抓取生成合成照片和表达。基于Blender和BlenderProc。

示例:  
![fig1](/images/fig1.jpg "fig1")
![fig2](/images/fig2.jpg "fig2")
![fig3](/images/fig3.png "fig3")

建议为项目使用conda环境。以下是设置项目环境并运行数据集生成过程的指南。项目在**Windows 10**和**Ubuntu 18.04**上进行了测试。
# 快速开始
<!--
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
python -m blenderproc pip install pandas tqdm debugpy openpyxl
```
### 5. Download models and scene:
Download from [Google Drive](https://drive.google.com/file/d/1tDshqp_SNw9SoH4jtoeFkZu8dhrZSY12/view?usp=sharing) or [Quark Drive](https://pan.quark.cn/s/d94083a37db0).  
Extract the downloaded file to the project directory. The directory structure should look like this:
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
-->
### 1. 克隆项目:
```bash
cd /path/to/project
mkdir RefGrasp && cd RefGrasp
git clone https://github.com/Soappyooo/Refer_Grasp.git .
```
### 2. 创建conda环境:
```bash
conda create -n refgrasp python==3.10.13
conda activate refgrasp
pip install -r requirements.txt
```
### 3. 安装BlenderProc:
```bash
pip install -e ./BlenderProc
```
### 4. 安装所需包:
可能需要一段时间来安装包，因为首先会安装Blender 3.5。
```bash
python -m blenderproc pip install pandas tqdm debugpy openpyxl
```
### 5. 下载模型和场景:
从[Google Drive](https://drive.google.com/file/d/1tDshqp_SNw9SoH4jtoeFkZu8dhrZSY12/view?usp=sharing)或[夸克网盘](https://pan.quark.cn/s/d94083a37db0)下载(760MB)。
将下载的文件解压到项目目录。目录结构应如下所示:
<pre>
/RefGrasp  
├─ <b>blender_files</b>
│  └─ <b>background.blend</b>
├─ BlenderProc  
├─ core  
├─ images 
├─ <b>models</b> 
│  └─ <b>...(290个文件)</b>
├─ utils
└─ ...(其它文件)  
</pre>
### 6. 运行 `dataset_generation.py`:
运行脚本将在`output_demo`目录中生成一个演示数据集(约50张图片)。
```bash
python -m blenderproc run dataset_generation.py
```
<br>

# 数据集生成
### 在单个GPU上使用参数运行:
例如，要生成一个包含1000次迭代和每次迭代5张图片的数据集，请运行:
```bash
python -m blenderproc run dataset_generation.py --output-dir ./output --iterations 1000 --images-per-iteration 5
```
#### 所有可能的参数如下:  
`--gpu-id`: 用于渲染的GPU设备id。默认为0。  
`--output-dir`: 生成数据集的输出目录。默认为"./output_demo"。  
`--obj-dir`: 模型文件目录。默认为"./models"。  
`--blender-scene-file-path`: 背景场景文件路径。默认为"./blender_files/background.blend"。  
`--models-info-file-path`: 模型信息文件路径。默认为"./models/models_info_all.xlsx"。  
`--log-file-path`: 日志文件路径。默认为"./output_demo/dataset_generation.log"。  
`--iterations`: 迭代次数。默认为10。  
`--images-per-iteration`: 每次迭代的图片数量。总图片数 <= 迭代次数 * 每次迭代的图片数量。同一个迭代中的图片共享相同的场景，但具有不同的背景、光照和相机姿势。默认为5。  
`--resolution-width`: 渲染分辨率宽度。默认为512。  
`--resolution-height`: 渲染分辨率高度。默认为512。  
`--scene-graph-rows`: 场景图行数。默认为4。  
`--scene-graph-cols`: 场景图列数。默认为4。  
`--seed`: 随机种子，除非调试，否则不应设置。默认为None。  
`--persistent-data-cleanup-interval`: 清理持久数据间隔。默认为7。  
`--texture-limit`: 纹理限制。默认为"2048"。  
`--cpu-threads`: 渲染中使用的CPU线程数。0为自动设置。默认为0。

### 在多个GPU上使用参数运行:
例如，要在4个GPU上生成一个包含1000次迭代和每次迭代5张图片的数据集，请运行:
```bash
python dataset_generation_multi_gpu.py --gpu-ids 0,1,2,3 --output-dir ./output --iteration 1000 --images-per-iteration 5
```
#### 所有可能的参数如下: 
`--gpu-ids`: 用于渲染的GPU设备id。默认为"0,1,2,3"。  
`--output-dir`: 生成数据集的输出目录。默认为"./output"。  
`--obj-dir`: 模型文件目录。默认为"./models"。  
`--blender-scene-file-path`: 背景场景文件路径。默认为"./blender_files/background.blend"。  
`--models-info-file-path`: 模型信息文件路径。默认为"./models/models_info_all.xlsx"。  
`--iterations`: 每个GPU的迭代次数。默认为10。  
`--images-per-iteration`: 每次迭代的图片数量。总图片数 <= 迭代次数 * 每次迭代的图片数量 * GPU数量。默认为5。  
`--resolution-width`: 渲染分辨率宽度。默认为512。  
`--resolution-height`: 渲染分辨率高度。默认为512。  
`--scene-graph-rows`: 场景图行数。默认为4。  
`--scene-graph-cols`: 场景图列数。默认为4。  
`--persistent-data-cleanup-interval`: 清理持久数据间隔。默认为7。  
`--texture-limit`: 纹理限制。默认为"2048"。  
`--cpu-threads`: 每个子进程中用于渲染的CPU线程数。0为自动设置。默认为16。

[返回顶部](#top)