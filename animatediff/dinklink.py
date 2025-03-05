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
    create_dinklink()
    prepare_dinklink()

def create_dinklink():
    if not hasattr(comfy.hooks, DINKLINK):
        setattr(comfy.hooks, DINKLINK, {})

def get_dinklink() -> dict[str, dict[str]]:
    create_dinklink()
    return getattr(comfy.hooks, DINKLINK)

class DinkLinkConst:
    VERSION = "version"
    # ACN
    ACN = "ACN"
    ACN_CREATE_OUTER_SAMPLE_WRAPPER = "create_outer_sample_wrapper"
    # ADE
    ADE = "ADE"
    ADE_ANIMATEDIFFMODEL = "AnimateDiffModel"
    ADE_ANIMATEDIFFINFO = "AnimateDiffInfo"
    ADE_CREATE_MOTIONMODELPATCHER = "create_MotionModelPatcher"

def prepare_dinklink():
    # expose classes
    d = get_dinklink()
    link_ade = d.setdefault(DinkLinkConst.ADE, {})
    link_ade[DinkLinkConst.VERSION] = 10000

def get_acn_outer_sample_wrapper(throw_exception=True):
    d = get_dinklink()
    try:
        link_acn = d[DinkLinkConst.ACN]
        return link_acn[DinkLinkConst.ACN_CREATE_OUTER_SAMPLE_WRAPPER]
    except KeyError:
        if throw_exception:
            raise Exception("Advanced-ControlNet nodes need to be installed to make use of ContextRef; " + \
                            "they are either not installed or are of an insufficient version.")
    return None

def get_acn_dinklink_version(throw_exception=True):
    d = get_dinklink()
    try:
        link_acn = d[DinkLinkConst.ACN]
        return link_acn[DinkLinkConst.VERSION]
    except KeyError:
        if throw_exception:
            raise Exception("Advanced-ControlNet nodes need to be installed to make use of ContextRef; " + \
                            "they are either not installed or are of an insufficient version.")
    return None
