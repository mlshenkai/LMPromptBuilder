# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/7/6 9:30 PM
# @File: convert_llava_instruct_150k_images
# @Email: mlshenkai@163.com
from pathlib import Path
from tqdm import tqdm
import shutil


def convert_llava_instruct_150k_images(
    coco_image_dir: Path = Path("/code-online/resources/data/mscoco/data/train2014"),
    llava_instruct_150k_images_dir: Path = Path(
        "/code-online/resources/data/llava_instruct_150k/data"
    ),
):
    coco_images_path_list = coco_image_dir.glob("*.jpg")
    for coco_images_path in tqdm(coco_images_path_list):
        file_name = coco_images_path.stem
        # file_name = file_name.split("_")[-1]
        print(file_name)
        shutil.copy(
            coco_images_path.as_posix(),
            llava_instruct_150k_images_dir.joinpath(f"{file_name}.jpg").as_posix(),
        )


if __name__ == "__main__":
    convert_llava_instruct_150k_images()
