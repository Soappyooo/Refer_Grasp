import subprocess
import argparse
import os
import logging
import signal

# set parameters
GPU_IDS = "0,1,2,3"  # GPU devices id used for rendering
OBJ_DIR = os.path.abspath("./models")  # obj files directory
BLENDER_SCENE_FILE_PATH = os.path.abspath("./blender_files/background.blend")  # background scene file path
OUTPUT_DIR = os.path.abspath("./output1")
MODELS_INFO_FILE_PATH = os.path.abspath("./models/models_info_all.xlsx")
LOG_FILE_PATH = os.path.abspath(os.path.join(OUTPUT_DIR, "./dataset_generation_gather.log"))
ITERATIONS = 5
IMAGES_PER_ITERATION = 5  # total images <= IMAGES_PER_ITERATION * ITERATIONS
RESOLUTION_WIDTH = 512
RESOLUTION_HEIGHT = 512
SCENE_GRAPH_ROWS = 4
SCENE_GRAPH_COLS = 4
SEED = None
PERSISITENT_DATA_CLEANUP_INTERVAL = 7  # clean up persistent data may speed up rendering for large dataset generation
TEXTURE_LIMIT = "2048"

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-ids", type=str, default=GPU_IDS, help="gpu device ids to use")
parser.add_argument("--obj-dir", type=str, default=OBJ_DIR, help="obj files directory")
parser.add_argument("--blender-scene-file-path", type=str, default=BLENDER_SCENE_FILE_PATH, help="background scene file path")
parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="output directory")
parser.add_argument("--models-info-file-path", type=str, default=MODELS_INFO_FILE_PATH, help="models info file path")
parser.add_argument("--log-file-path", type=str, default=LOG_FILE_PATH, help="log file path")
parser.add_argument("--iterations", type=int, default=ITERATIONS, help="number of iterations per gpu")
parser.add_argument("--images-per-iteration", type=int, default=IMAGES_PER_ITERATION, help="total images <= iterations * images_per_iteration")
parser.add_argument("--resolution-width", type=int, default=RESOLUTION_WIDTH, help="render resolution width")
parser.add_argument("--resolution-height", type=int, default=RESOLUTION_HEIGHT, help="render resolution height")
parser.add_argument("--scene-graph-rows", type=int, default=SCENE_GRAPH_ROWS, help="scene graph rows")
parser.add_argument("--scene-graph-cols", type=int, default=SCENE_GRAPH_COLS, help="scene graph cols")
parser.add_argument(
    "--persistent-data-cleanup-interval", type=int, default=PERSISITENT_DATA_CLEANUP_INTERVAL, help="clean up persistent data interval"
)
parser.add_argument("--texture-limit", type=str, default=TEXTURE_LIMIT, help="texture limit")
args = parser.parse_args()
gpu_idxs: list[int] = list(map(int, args.gpu_ids.split(",")))

# configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(args.log_file_path, mode="w", encoding="utf-8")
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

# start subprocesses
processes = []
log_file_paths = []
for gpu_idx in gpu_idxs:
    output_dir_per_gpu = os.path.abspath(os.path.join(args.output_dir, f"gpu_{gpu_idx}"))
    log_file_path_per_gpu = os.path.abspath(os.path.join(output_dir_per_gpu, "dataset_generation.log"))
    log_file_paths.append(log_file_path_per_gpu)
    process = subprocess.Popen(
        f"python -m blenderproc run dataset_generation.py --gpu-id {gpu_idx} --obj-dir {args.obj_dir} "
        f"--blender-scene-file-path {args.blender_scene_file_path} --output-dir {output_dir_per_gpu} "
        f"--models-info-file-path {args.models_info_file_path} --log-file-path {log_file_path_per_gpu} "
        f"--iterations {args.iterations} --images-per-iteration {args.images_per_iteration} "
        f"--resolution-width {args.resolution_width} --resolution-height {args.resolution_height} "
        f"--scene-graph-rows {args.scene_graph_rows} --scene-graph-cols {args.scene_graph_cols} "
        f"--persistent-data-cleanup-interval {args.persistent_data_cleanup_interval} "
        f"--texture-limit {args.texture_limit}",
        shell=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    processes.append(process)
    logger.info(f"PID: {process.pid}, GPU {gpu_idx}: started")


# terminate subprocesses when receiving SIGINT or SIGTERM
def signal_handler(sig, frame):
    for process in processes:
        process.terminate()
        process.wait()
    logger.error(f"Received signal {sig}, all subprocesses terminated")
    exit(1)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    while any([process.poll() is None for process in processes]):
        for i, process in enumerate(processes):
            if process.poll() is not None and process.poll() != 0:
                raise Exception(f"PID: {process.pid} exited with code {process.poll()}, check {log_file_paths[i]} for more information")
            output = process.stdout.readline()
            if "Progress" in output and "INFO" in output:
                logger.info(f"PID: {process.pid}, GPU {gpu_idxs[i]}: {output.rstrip()}")
except Exception as e:
    for process in processes:
        process.terminate()
        process.wait()
    logger.error(e)
    raise e

logger.info("All subprocesses finished")