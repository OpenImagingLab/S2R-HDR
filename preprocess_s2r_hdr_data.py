"""
1. Removing the lowest 1% and highest 1% of the original data and re-normalizing it, primarily to reduce noise caused by rendering.
2. Cropping the image from 1920x1080 to 640x640 to speed up the training process.
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import numpy as np
import cv2
import os
import shutil
from tqdm import *
import multiprocessing
import time
import pyexr

# TODO  you can modify this file
SRC_DIR = "./data/S2R-HDR" # original S2R-HDR data path
PROCESSED_DIR = "./data/S2R-HDR-processed" # S2R-HDR processed path
PROCESSED_PATCH_DIR = "./data/S2R-HDR-processed-patch" # S2R-HDR processed patch path 

def read_hdr(image_path):
    img = cv2.imread(image_path, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.clip(img, 0.0, float("inf"))
    return img


def preprocess_data_range(img, minn, maxx):
    img = img.clip(minn, maxx)
    img = (img - minn) / (maxx - minn)
    img = img.clip(0.0, 1.0)
    return img


def create_hdr_data_pyexr(i):
    import pyexr
    begin_frame = 3
    end_frame = 3 + 24
    scene_list = []
    for scene in os.listdir(SRC_DIR):
        if os.path.isfile(os.path.join(SRC_DIR, scene)):
            continue
        scene_list.append(scene)
    # for scene in scene_list:
    for scene in (scene_list[i], ):
        if os.path.isfile(os.path.join(SRC_DIR, scene)):
            continue
        scene_dir = os.path.join(SRC_DIR, scene)
        save_path = os.path.join(PROCESSED_DIR, scene, "img")
        os.makedirs(save_path, exist_ok=True)
        hdr_list = []
        for i in range(begin_frame, end_frame):
            hdr_list.append(read_hdr(os.path.join(scene_dir, "img", "HDR", f"{i:04d}.exr")))
        first = hdr_list[0]
        minn = np.percentile(first, 1)
        maxx = np.percentile(first, 99)
        for i, hdr in enumerate(hdr_list):
            hdr = preprocess_data_range(hdr, minn, maxx)
            # shutil.copyfile(os.path.join(scene_dir, "img", f"{i:04d}.exr")), os.path.join(save_path, f"{i:04d}.exr"))
            pyexr.write(os.path.join(save_path, f"{i:04d}.exr"), 
                        hdr.astype(np.float16), channel_names=['R','G','B'], precision=pyexr.HALF) #  compression=pyexr.ZIP_COMPRESSION
        # break


def crop_dataset_pyexr(idx, patch_size=640, stride=512):
    import pyexr

    scene_list = []
    for scene in os.listdir(PROCESSED_DIR):
        if os.path.isfile(os.path.join(PROCESSED_DIR, scene)):
            continue
        scene_list.append(scene)
    # for scene in tqdm(scene_list):
    for scene in (scene_list[idx], ):
        if os.path.isfile(os.path.join(PROCESSED_DIR, scene)):
            continue
        hdr_list = []
        for i in range(0, 24):
            with pyexr.open(os.path.join(PROCESSED_DIR, scene, "img", f"{i:04d}.exr")) as file:
                img = file.get(precision=pyexr.HALF)
            hdr_list.append(img.copy())
        h, w, _ = hdr_list[0].shape # 1920 1080
        # count = 0
        for x in range(0, w, stride):
            for y in range(0, h, stride):
                if x + 256 > w or y + 256 > h:
                    continue
                # print(idx, x, y)
                save_path = os.path.join(PROCESSED_PATCH_DIR, scene + f"_{x}_{y}", "img")
                os.makedirs(save_path, exist_ok=True)
                if x + patch_size > w:
                    x = w - patch_size
                if y + patch_size > h:
                    y = h - patch_size
                for i, hdr in enumerate(hdr_list):
                    crop_hdr = hdr[y:y + patch_size, x:x + patch_size]
                    pyexr.write(os.path.join(save_path, f"{i:04d}.exr"), 
                                crop_hdr.astype(np.float16), channel_names=['R','G','B'], precision=pyexr.HALF) #  compression=pyexr.ZIP_COMPRESSION
                # count += 1
        # print(count)
        # break

def get_scene_length(root_dir):
    scene_list = []
    for scene in os.listdir(root_dir):
        if os.path.isfile(os.path.join(root_dir, scene)):
            continue
        scene_list.append(scene)
    return len(scene_list)

def check(check_dir=PROCESSED_PATCH_DIR):
    scene_list = []
    for scene in os.listdir(check_dir):
        if os.path.isfile(os.path.join(check_dir, scene)):
            continue
        scene_list.append(scene)
    
    for scene in tqdm(scene_list):
        for i in range(24):
            try:
                cv2.imread(os.path.join(check_dir, scene, "img",f"{i:04d}.exr"), -1)
            except Exception:
                print(scene, f"{i:04d}.exr")


def create_scene_list(check_dir=PROCESSED_PATCH_DIR):
    scene_list = []
    for scene in os.listdir(check_dir):
        if os.path.isfile(os.path.join(check_dir, scene)):
            continue
        scene_list.append(scene)
    with open(os.path.join(check_dir, "trainlist.txt"), "w+") as f:
        for i, scene in enumerate(scene_list):
            if i == 0:
                f.write(f"{scene}")
            else:
                f.write(f"\n{scene}")
    print(f"write {check_dir}/trainlist.txt successful")


if __name__ == "__main__":
    # 1. processing data
    print("Process data....")
    # create_hdr_data_pyexr(0)
    pool = multiprocessing.Pool(processes=16)
    pool.map(create_hdr_data_pyexr, range(0, get_scene_length(SRC_DIR)))
    pool.close()
    pool.join()
    create_scene_list(PROCESSED_DIR)

    # 2. crop dataset
    print("Crop data....")
    # crop_dataset_pyexr(0)
    pool = multiprocessing.Pool(processes=16)
    pool.map(crop_dataset_pyexr, range(0, get_scene_length(PROCESSED_DIR)))
    pool.close()
    pool.join()
    create_scene_list(PROCESSED_PATCH_DIR)

    # check
    print("Check data....")
    # check() # you can check processing data, but it is not necessary.
    print("successful")