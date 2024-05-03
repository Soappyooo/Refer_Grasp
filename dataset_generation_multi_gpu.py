import subprocess
import argparse
import os
import logging
import signal
import sys
from utils.dataset_utils import DatasetUtils


# set parameters
GPU_IDS = "0,1,2,3"  # GPU devices id used for rendering
OBJ_DIR = os.path.abspath("./models")  # obj files directory
BLENDER_SCENE_FILE_PATH = os.path.abspath("./blender_files/background.blend")  # background scene file path
OUTPUT_DIR = os.path.abspath("./output")
MODELS_INFO_FILE_PATH = os.path.abspath("./models/models_info_all.xlsx")
# LOG_FILE_PATH = os.path.abspath(os.path.join(OUTPUT_DIR, "./dataset_generation_gather.log"))
ITERATIONS = 10
IMAGES_PER_ITERATION = 5  # total images <= IMAGES_PER_ITERATION * ITERATIONS
RESOLUTION_WIDTH = 512
RESOLUTION_HEIGHT = 512
SCENE_GRAPH_ROWS = 4
SCENE_GRAPH_COLS = 4
SEED = None
PERSISITENT_DATA_CLEANUP_INTERVAL = 100  # clean up persistent data may speed up rendering for large dataset generation
TEXTURE_LIMIT = "2048"
CPU_THREADS = 16
SAMPLES = 512

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-ids", type=str, default=GPU_IDS, help="GPU device ids to use")
parser.add_argument("--obj-dir", type=str, default=OBJ_DIR, help="obj files directory")
parser.add_argument("--blender-scene-file-path", type=str, default=BLENDER_SCENE_FILE_PATH, help="background scene file path")
parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="output directory")
parser.add_argument("--models-info-file-path", type=str, default=MODELS_INFO_FILE_PATH, help="models info file path")
# parser.add_argument("--log-file-path", type=str, default=LOG_FILE_PATH, help="log file path")
parser.add_argument("--iterations", type=int, default=ITERATIONS, help="number of iterations per GPU")
parser.add_argument("--images-per-iteration", type=int, default=IMAGES_PER_ITERATION, help="total images <= iterations * images_per_iteration")
parser.add_argument("--resolution-width", type=int, default=RESOLUTION_WIDTH, help="render resolution width")
parser.add_argument("--resolution-height", type=int, default=RESOLUTION_HEIGHT, help="render resolution height")
parser.add_argument("--scene-graph-rows", type=int, default=SCENE_GRAPH_ROWS, help="scene graph rows")
parser.add_argument("--scene-graph-cols", type=int, default=SCENE_GRAPH_COLS, help="scene graph cols")
parser.add_argument(
    "--persistent-data-cleanup-interval", type=int, default=PERSISITENT_DATA_CLEANUP_INTERVAL, help="clean up persistent data interval"
)
parser.add_argument("--texture-limit", type=str, default=TEXTURE_LIMIT, help="texture limit")
parser.add_argument("--cpu-threads", type=int, default=CPU_THREADS, help="number of CPU threads used in render for each subprocess")
parser.add_argument("--samples", type=int, default=SAMPLES, help="max samples for rendering")
args = parser.parse_args()
gpu_idxs: list[int] = list(map(int, args.gpu_ids.split(",")))
log_file_path_gather = os.path.abspath(os.path.join(args.output_dir, "dataset_generation_gather.log"))
if not os.path.exists(os.path.dirname(log_file_path_gather)):
    os.makedirs(os.path.dirname(log_file_path_gather), exist_ok=True)
# configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_file_path_gather, mode="w", encoding="utf-8")
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)
logger.info(f"Log file path: {log_file_path_gather}")


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


# start subprocesses
processes = []
log_file_paths = []
for gpu_idx in gpu_idxs:
    output_dir_per_gpu = os.path.abspath(os.path.join(args.output_dir, "temp", f"gpu_{gpu_idx}"))
    log_file_path_per_gpu = os.path.abspath(os.path.join(output_dir_per_gpu, "dataset_generation.log"))
    log_file_paths.append(log_file_path_per_gpu)
    process = subprocess.Popen(
        f"python -m blenderproc run dataset_generation.py --gpu-id {gpu_idx} --obj-dir {args.obj_dir} "
        f"--blender-scene-file-path {args.blender_scene_file_path} --output-dir {output_dir_per_gpu} "
        f"--models-info-file-path {args.models_info_file_path} --log-file-path {log_file_path_per_gpu} "
        f"--iterations {args.iterations} --images-per-iteration {args.images_per_iteration} "
        f"--resolution-width {args.resolution_width} --resolution-height {args.resolution_height} "
        f"--scene-graph-rows {args.scene_graph_rows} --scene-graph-cols {args.scene_graph_cols} "
        f"--persistent-data-cleanup-interval {args.persistent_data_cleanup_interval} --samples {args.samples} "
        f"--texture-limit {args.texture_limit} --cpu-threads {args.cpu_threads}",
        shell=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    processes.append(process)
    os.set_blocking(process.stdout.fileno(), False) if os.name == "posix" else None  # windows may have blocking issue
    logger.info(f"PID: {process.pid}, GPU {gpu_idx}: started")


# terminate subprocesses when receiving SIGINT or SIGTERM
def signal_handler(sig, frame):
    for process in processes:
        os.kill(process.pid, signal.SIGINT)
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
                logger.info(f"PID: {process.pid}, GPU {gpu_idxs[i]}: {output.rstrip().split('-')[-1]}")
                sys.stdout.flush()
except Exception as e:
    for process in processes:
        process.terminate()
        process.wait()
    logger.error(e)
    raise e

logger.info("All subprocesses finished, start merging dataset...")

DatasetUtils.merge_datasets_from_multi_gpus([os.path.join(args.output_dir, "temp", f"gpu_{gpu_idx}") for gpu_idx in gpu_idxs], args.output_dir)

logger.info("Dataset merged successfully, exit")
