import os, sys

os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.getcwd())

from utils.dataset_utils import DatasetUtils
import multiprocessing
from functools import partial
from tqdm import tqdm


root_path = "./output"
process_count = 32


def process_image(i, root_path, expression_temp):
    DatasetUtils.visualize_image(
        os.path.join(root_path, "rgb", f"rgb_{i:08d}.jpg"),
        os.path.join(root_path, "mask", f"mask_{i:08d}.png"),
        os.path.join(root_path, "expressions.json"),
        save_image=True,
        save_path=os.path.join(root_path, "annotations"),
        show_image=False,
        expression_json_temp=expression_temp,
    )


if __name__ == "__main__":
    expression_temp = []
    num_images = len(os.listdir(os.path.join(root_path, "rgb")))
    with multiprocessing.Pool(process_count) as pool:
        for _ in tqdm(pool.imap(partial(process_image, root_path=root_path, expression_temp=expression_temp), range(num_images)), total=num_images):
            pass
