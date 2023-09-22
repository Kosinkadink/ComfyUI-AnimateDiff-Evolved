# AnimateDiff for ComfyUI

Improved [AnimateDiff](https://github.com/guoyww/AnimateDiff/) integration for ComfyUI, initially adapted from [sd-webui-animatediff](https://github.com/continue-revolution/sd-webui-animatediff) but changed greatly since then. Please read the AnimateDiff repo README for more information about how it works at its core.

Examples shown here will also often make use of two helpful set of nodes:
- [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet) for loading files in batches and controlling which latents should be affected by the ControlNet inputs (work in progress, will include more advance workflows + features for AnimateDiff usage later).
- [comfy_controlnet_preprocessors](https://github.com/Fannovel16/comfy_controlnet_preprocessors) for ControlNet preprocessors not present in vanilla ComfyUI; this repo is archived, and future development by the dev will happen here: [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux). While most preprocessors are common between the two, some give different results. Workflows linked here use the archived version, comfy_controlnet_preprocessors. (TODO: I'll reinvestigate with more recent changes and update as needed)

# Installation

## If using Comfy Manager:

1. Look for ```AnimateDiff```, and be sure it is ```(Kosinkadink version)```. Install it.
![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/10324120-aaee-460a-8ec9-57a46c1b9edc)


## If installing manually:
1. Clone this repo into `custom_nodes` folder.

# How to Use:
1. Download motion modules. You will need at least 1. Different modules produce different results.
   - Original models ```mm_sd_v14```, ```mm_sd_v15```, and ```mm_sd_v15_v2```: [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI) | [HuggingFace](https://huggingface.co/guoyww/animatediff) | [CivitAI](https://civitai.com/models/108836) | [Baidu NetDisk](https://pan.baidu.com/s/18ZpcSM6poBqxWNHtnyMcxg?pwd=et8y).
   - Stabilized finetunes of mm_sd_v14, ```mm-Stabilized_mid``` and ```mm-Stabilized_high```, by **manshoety**: [HuggingFace](https://huggingface.co/manshoety/AD_Stabilized_Motion/tree/main)
   - Higher resolution finetune,```temporaldiff-v1-animatediff```  by **CiaraRowles**: [HuggingFace](https://huggingface.co/CiaraRowles/TemporalDiff/tree/main)
2. Place models in ```ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/models```. They can be renamed if you want.
3. Get creative! If it works for normal image generation, it (probably) will work for AnimateDiff generations. Latent upscales? Go for it. ControlNets, one or more stacked? You betcha. Masking the conditioning of ControlNets to only affect part of the animation? Sure. Try stuff and you will be surprised by what you can do. Samples with workflows are included below.

# Features:
- Compatible with a variety of samplers, vanilla KSampler nodes and KSampler (Effiecient) nodes.
- ControlNet support - both per-frame, or "interpolating" between frames; can kind of use this as img2video (see workflows below)
- Infinite animation length support using sliding context windows (introduced 9/17/23)

# Upcoming features:
- Prompt travel, and in general more control over per-frame conditioning
- Alternate context schedulers and context types

# Core Nodes:

## AnimateDiff Loader
![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/eb07560b-5270-4fc0-9016-311a0327c413)

The ***only required node to use AnimateDiff***, the Loader outputs a model that will perform AnimateDiff functionality when passed into a sampling node.

Inputs:
- model: model to setup for AnimateDiff usage. ***Must be a SD1.5-derived model.***
- context_options: optional context window to use while sampling; if passed in, total animation length has no limit. If not passed in, animation length will be limited to either 24 or 32 frames, depending on motion model.
- model_name: motion model to use with AnimateDiff.
- beta_schedule: noise scheduler for SD. ```sqrt_linear``` is the intended way to use AnimateDiff, with expected saturation. However, ```linear``` can give useful results as well, so feel free to experiment.

Outputs:
- MODEL: model injected to perform AnimateDiff functions

### Usage
To use, just plug in your model into the AnimateDiff Loader. When the output model (and any derivative of it in this pathway) is passed into a sampling node, AnimateDiff will do its thing.

The desired animation length is determined by the latents passed into the sampler. **With context_options connected, there is no limit to the amount of latents you can pass in, AKA unlimited animation length.** When no context_options are connected, the sweetspot is 16 latents passed in for best results, with a limit of 24 or 32 based on motion model loaded. **These same rules apply to Uniform Context Option's context_length**.
![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/08cc9da9-a21c-469b-8ed6-6153845f80b9)
![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/7d9a21aa-59ee-47ec-8949-8f6a746e7bd7)


## Uniform Context Options
TODO: fill this out


## Samples (download or drag images of the workflows into ComfyUI to instantly load the corresponding workflows!)

### txt2img

![t2i_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/b2a86e3f-1eaf-4609-8c29-8226c32985fe)

![aaa_readme_00001_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/adf2d591-85c4-4d84-9a6f-f7296b5b7f76)

[aaa_readme_00003_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/974c77ea-47a2-422f-bea5-b080549fb17c)



### txt2img - 48 frame animation with 16 context_length (uniform)

![t2i_context_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/8dd62a63-0907-4691-9964-37c2d5eb226f)

![aaa_readme_00017_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/3033dc45-2876-4d14-9546-ab59a00d8ca9)

[aaa_readme_00018_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/9b3b5d7d-07da-4b5c-80bc-b3cd82475c71)



### txt2img w/ latent upscale (partial denoise on upscale)

![txt2image_upscale_partialdenoise_workflow](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/82a9ba7b-6da1-4eee-bead-8aca55943fb9)

![AA_upscale_gif_00007_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/0cb2ca7e-8666-4abc-86f2-f24a20ff4bed)


### txt2img w/ latent upscale (partial denoise on upscale) - 48 frame animation with 16 frame window

![txt2image_sliding_upscale_partialdenoise_workflow](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/9980e9dc-dde3-421f-8c90-319b71616fdc)

TODO: add generated image here (gif is too big for github)


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



## Known Issues

### Some motion models have visible watermark on resulting images (especially when using mm_sd_v15)

Training data used by the authors of the AnimateDiff paper contained Shutterstock watermarks. Since mm_sd_v15 was finetuned on finer, less drastic movement, the motion module attempts to replicate the transparency of that watermark and does not get blurred away like mm_sd_v14. Using other motion modules, or combinations of them using Advanced KSamplers should alleviate watermark issues.
