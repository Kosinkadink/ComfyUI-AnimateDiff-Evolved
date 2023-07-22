import os
from huggingface_hub import hf_hub_download


HF_REPO = "guoyww/animatediff"
MODEL_FILES = ["mm_sd_v14.ckpt", "mm_sd_v15.ckpt"]

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../models"))


def get_available_models():
    available_model = [
        f for f in MODEL_FILES if os.path.exists(os.path.join(MODEL_DIR, f))
    ]

    return available_model


def download(model_file=MODEL_FILES[-1]):
    if not os.path.exists(os.path.join(MODEL_DIR, model_file)):
        hf_hub_download(
            HF_REPO,
            model_file,
            cache_dir=MODEL_DIR,
            force_download=True,
            force_filename=model_file,
        )
