####################################################################################################
# DinkLink is my method of sharing classes/functions between my nodes.
#
# My DinkLink-compatible nodes will inject comfy.hooks with a __DINKLINK attr
# that stores a dictionary, where any of my node packs can store their stuff.
#
# It is not intended to be accessed by node packs that I don't develop, so things may change
# at any time.
#
# DinkLink also serves as a proof-of-concept for a future ComfyUI implementation of
# purposely exposing node pack classes/functions with other node packs.
####################################################################################################
from __future__ import annotations
import comfy.hooks

DINKLINK = "__DINKLINK"

def init_dinklink():
    if not hasattr(comfy.hooks, DINKLINK):
        setattr(comfy.hooks, DINKLINK, {})
    prepare_dinklink()


def get_dinklink() -> dict[str, dict[str]]:
    getattr(comfy.hooks, DINKLINK)


def prepare_dinklink():
    pass
