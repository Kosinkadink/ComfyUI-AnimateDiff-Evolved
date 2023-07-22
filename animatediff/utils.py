#このコードはhttps://github.com/kohya-ss/sd-scripts/blob/main/finetune/tag_images_by_wd14_tagger.pyを参考にしていますというかパクっています。

from huggingface_hub import hf_hub_download
import cv2
import numpy as np
import os
IMAGE_SIZE = 448

TAGGER_REPO = "furusu/wd-v1-4-tagger-pytorch"
TAGGER_FILE = "wd-v1-4-vit-tagger-v2.ckpt"

def download(path):
    if not os.path.exists(os.path.join(path, TAGGER_FILE)):
        hf_hub_download(TAGGER_REPO, TAGGER_FILE, cache_dir=path, force_download=True, force_filename=TAGGER_FILE)
