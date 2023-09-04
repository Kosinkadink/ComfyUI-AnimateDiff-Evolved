from .animatediff.logger import logger
from .animatediff.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .animatediff.model_utils import get_available_models, get_folder_path, Folders

if len(get_available_models()) == 0:
    logger.error(f"No motion models found. Please download one and place in: {get_folder_path(Folders.MODELS)}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
