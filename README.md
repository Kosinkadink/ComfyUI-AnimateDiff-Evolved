# AnimateDiff for ComfyUI

Improved [AnimateDiff](https://github.com/guoyww/AnimateDiff/) integration for ComfyUI, initially adapted from [sd-webui-animatediff](https://github.com/continue-revolution/sd-webui-animatediff) but changed greatly since then. Please read the AnimateDiff repo README for more information about how it works at its core.

Examples shown here will also often make use of two helpful set of nodes:
- [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet) for loading files in batches and controlling which latents should be affected by the ControlNet inputs (work in progress, will include more advance workflows + features for AnimateDiff usage later).
- [comfy_controlnet_preprocessors](https://github.com/Fannovel16/comfy_controlnet_preprocessors) for ControlNet preprocessors not present in vanilla ComfyUI; this repo is archived, and future development by the dev will happen here: [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux). While most preprocessors are common between the two, some give different results. Workflows linked here use the archived version, comfy_controlnet_preprocessors.

## Installation

🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥

***IMPORTANT: if you already have ArtVentureX's version of AnimateDiff installed, either remove the ```custom_nodes/comfyui-animatediff``` folder, uninstall or disable it using ComfyUI-Manager, or add .disabled to the end of that folder's name! Otherwise things will go wrong!***

🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥

### If using ComfyUI-Manager:

1. Look for ```AnimateDiff```, and be sure it is ```(Kosinkadink version)```. Install it.

![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/10324120-aaee-460a-8ec9-57a46c1b9edc)


### If installing manually:
1. Clone this repo into `custom_nodes` folder.

## How to Use:
1. Download motion modules from [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI) | [HuggingFace](https://huggingface.co/guoyww/animatediff) | [CivitAI](https://civitai.com/models/108836) | [Baidu NetDisk](https://pan.baidu.com/s/18ZpcSM6poBqxWNHtnyMcxg?pwd=et8y). You can download one or more motion models. Place models in ComfyUI/custom_nodes/ComfyUI-AnimateDiff/models. They can be renamed if you want. More motion modules are being trained by the community - if I am made aware of any good ones, I will link here as well. (TODO: create .safetensor versions of the motion modules and share them here.)
2. Get creative! If it works for normal image generation, it (probably) will work for AnimateDiff generations. Latent upscales? Go for it. ControlNets, one or more stacked? You betcha. Masking the conditioning of ControlNets to only affect part of the animation? Sure. Try stuff and you will be surprised by what you can do. Samples with workflows are included below.

## Known Issues (and Solutions, please read!)

### Large resolutions may cause xformers to throw a CUDA error concerning a misconfigured value despite being within VRAM limitations.

It is an xformers bug accidentally triggered by the way the original AnimateDiff CrossAttention is passed in. Eventually either I will fix it, or xformers will. When encountered, the workaround is to boot ComfyUI with the "--disable-xformers" argument.

### GIF has Watermark (especially when using mm_sd_v15)

Training data used by the authors of the AnimateDiff paper contained Shutterstock watermarks. Since mm_sd_v15 was finetuned on finer, less drastic movement, the motion module attempts to replicate the transparency of that watermark and does not get blurred away like mm_sd_v14. Community finetunes of motion modules should eventually create equivalent (or better) results without the watermark. Until then, you'll need some good RNG or stick with mm_sd_v14, depending on your application.


## Samples (download or drag images of the workflows into ComfyUI to instantly load the corresponding workflows!)

### txt2img

![txt2image_workflow](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/2e6be92e-8f0b-428c-b7b4-84589ebd701e)

![AA_gif_00002_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/91933fb2-5b0b-4f41-a57a-ebebb604bd9d)


### txt2img w/ latent upscale (partial denoise on upscale)

![txt2image_upscale_partialdenoise_workflow](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/6a74ccb0-4c4a-4738-bf0f-8109c14507f1)

![AA_upscale_gif_00007_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/0cb2ca7e-8666-4abc-86f2-f24a20ff4bed)


### txt2img w/ latent upscale (full denoise on upscale)

![txt2image_upscale_workflow](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/ce9f4883-dc8f-47ff-976d-e73a65d7ba07)

![AA_upscale_gif_00001_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/4ca8abd2-0b48-41d6-9eea-ed1467a68f5f)


### txt2img w/ ControlNet-stabilized latent-upscale (partial denoise on upscale, Scaled Soft ControlNet Weights)

![txt2image_upscale_controlnetsoftweights_partialdenoise_workflow](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/44e6e761-d047-49c1-a05c-c809e6d215f5)

![AA_upscale_gif_00009_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/4f03d03d-839b-484d-b612-8add086a6b8b)


### txt2img w/ ControlNet-stabilized latent-upscale (full denoise on upscale)

![txt2image_upscale_controlnet_workflow](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/222555f4-3288-40e7-92dc-56329de4d816)

![AA_upscale_controlnet_gif_00006_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/480c9bed-132c-489c-9682-39856b87fedb)


### txt2img w/ Initial ControlNet input (using LineArt preprocessor on first txt2img as an example)

![txt2image_controlnet_workflow](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/9800f5bd-67e3-47e0-a7b1-739a06df4b76)

![AA_controlnet_gif_00017_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/de149c0f-bc1d-4bb9-8b4d-b10686e5b09f)


### txt2img w/ Initial ControlNet input (using OpenPose images) + latent upscale w/ full denoise

![txt2image_openpose_controlnet_upscale_workflow](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/328719b1-3b3e-4b53-819a-75b9436cc5e8)

(open_pose images provided courtesy of toyxyz)

![AA_openpose_cn_gif_00001_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/23291941-864d-495a-8ba8-d02e05756396)

![AA_gif_00029_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/8367b24e-dfe5-4942-8e21-ac5a562be731)


### img2img (TODO: this is outdated and still shows the old flickering version, update this)
<img width="1121" alt="Screenshot 2023-07-22 at 22 08 00" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/600f96b0-df21-4437-917f-7eda35ab6363">

![AnimateDiff_00002](https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/c78d64b9-b308-41ec-9804-bbde654d0b47)


## Upcoming features (aka TODO):
- Nodes for saving videos, saving generated files into a timestamped folder instead of all over ComfyUI output dir.
- Moving-window latent implementation for generating arbitrarily-long animations instead of being capped at 24 frames (moving window will still be limited to up to 24 frames).
- Add examples of using ControlNet to ease one image/controlnet input into another, and also add more nodes to Advanced-ControlNet to make it easier to do so
