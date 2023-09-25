# AnimateDiff for ComfyUI

Improved [AnimateDiff](https://github.com/guoyww/AnimateDiff/) integration for ComfyUI, initially adapted from [sd-webui-animatediff](https://github.com/continue-revolution/sd-webui-animatediff) but changed greatly since then. Please read the AnimateDiff repo README for more information about how it works at its core.

Examples shown here will also often make use of two helpful set of nodes:
- [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet) for loading files in batches and controlling which latents should be affected by the ControlNet inputs (work in progress, will include more advance workflows + features for AnimateDiff usage later).
- [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) for ControlNet preprocessors not present in vanilla ComfyUI. NOTE: If you previously used [comfy_controlnet_preprocessors](https://github.com/Fannovel16/comfy_controlnet_preprocessors), ***you will need to remove comfy_controlnet_preprocessors*** to avoid possible compatibility issues between the two. Actively maintained by Fannovel16.

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
   - Finetunes of mm_sd_v15_v2, ```mm-p_0.5.pth``` and ```mm-p_0.75.pth```, by **manshoety**: [HuggingFace](https://huggingface.co/manshoety/beta_testing_models/tree/main)
   - Higher resolution finetune,```temporaldiff-v1-animatediff```  by **CiaraRowles**: [HuggingFace](https://huggingface.co/CiaraRowles/TemporalDiff/tree/main)
2. Place models in ```ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/models```. They can be renamed if you want.
3. Optionally, you can use Motion LoRAs to influence movement of v2-based motion models like mm_sd_v15_v2.
   - [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI?usp=sharing) | [HuggingFace](https://huggingface.co/guoyww/animatediff) | [CivitAI](https://civitai.com/models/108836/animatediff-motion-modules)
   - Place Motion LoRAs in ```ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/motion-lora```. They can be renamed if you want.
5. Get creative! If it works for normal image generation, it (probably) will work for AnimateDiff generations. Latent upscales? Go for it. ControlNets, one or more stacked? You betcha. Masking the conditioning of ControlNets to only affect part of the animation? Sure. Try stuff and you will be surprised by what you can do. Samples with workflows are included below.


# Features:
- Compatible with a variety of samplers, vanilla KSampler nodes and KSampler (Effiecient) nodes.
- ControlNet support - both per-frame, or "interpolating" between frames; can kind of use this as img2video (see workflows below)
- Infinite animation length support using sliding context windows **(introduced 9/17/23)**
- Mixable Motion LoRAs from [original AnimateDiff repository](https://github.com/guoyww/animatediff/) implemented. Caveat: only really work on v2-based motion models like ```mm_sd_v15_v2```, ```mm-p_0.5.pth```, and ```mm-p_0.75.pth``` **(introduced 9/25/23)**

# Upcoming features:
- Prompt travel, and in general more control over per-frame conditioning (working on it now)
- Alternate context schedulers and context types

# Core Nodes:

## AnimateDiff Loader
![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/232ef170-30e0-4119-ace2-4cc9a842d1ac)


The ***only required node to use AnimateDiff***, the Loader outputs a model that will perform AnimateDiff functionality when passed into a sampling node.

Inputs:
- model: model to setup for AnimateDiff usage. ***Must be a SD1.5-derived model.***
- context_options: optional context window to use while sampling; if passed in, total animation length has no limit. If not passed in, animation length will be limited to either 24 or 32 frames, depending on motion model.
- motion_lora: optional motion LoRA input; if passed in, can influence movement.
- model_name: motion model to use with AnimateDiff.
- beta_schedule: noise scheduler for SD. ```sqrt_linear``` is the intended way to use AnimateDiff, with expected saturation. However, ```linear``` can give useful results as well, so feel free to experiment.

Outputs:
- MODEL: model injected to perform AnimateDiff functions

### Usage
To use, just plug in your model into the AnimateDiff Loader. When the output model (and any derivative of it in this pathway) is passed into a sampling node, AnimateDiff will do its thing.

The desired animation length is determined by the latents passed into the sampler. **With context_options connected, there is no limit to the amount of latents you can pass in, AKA unlimited animation length.** When no context_options are connected, the sweetspot is 16 latents passed in for best results, with a limit of 24 or 32 based on motion model loaded. **These same rules apply to Uniform Context Option's context_length**.

You can also connect AnimateDiff LoRA Loader nodes to influence the overall movement in the image - currently, only works well on motion model v2 like ss

![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/524ba030-97aa-47a5-a0fd-ecffbbf5e439)
![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/908d1848-13d4-4e86-bdb0-f6870fd28b06)



## Uniform Context Options
TODO: fill this out

## AnimateDiff LoRA Loader
![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/11159b61-7077-4cb1-864c-078bfe82ece3)

Allows plugging in Motion LoRAs into motion models. Current Motion LoRAs only properly support v2-based motion models. Does not affect sampling speed, as the values are frozen after model load. **If you experience slowdowns for using LoRAs, please open an issue so I can resolve it.**

Inputs:
- lora_name: name of Motion LoRAs placed in ```ComfyUI/custom_node/ComfyUI-AnimateDiff-Evolved/motion-lora``` directory.
- strength: how strong (or weak) effect of Motion LoRA should be. Too high a value can lead to artifacts in final render.
- prev_motion_lora: optional input allowing to stack LoRAs together.

Outputs:
- MOTION_LORA: motion_lora object storing the names of all the LoRAs that were chained behind it - can be plugged into the back of another AnimateDiff LoRA Loader, or into AniamateDiff Loader's motion_lora input.

![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/5e46261d-fe87-4daa-8ac3-3ef615f4619d)
![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/21ec6fa6-4874-4312-bd41-477307c9ebf8)



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

![t2i_lat_ups_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/6fc7acc0-337d-40c9-a7bd-3c37c0496ba0)

![aaa_readme_up_00001_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/f4199e25-c839-41ed-8986-fb7dbbe2ac52)

[aaa_readme_up_00002_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/2f44342f-3fd8-4863-8e3d-360377d608b7)



### txt2img w/ latent upscale (partial denoise on upscale) - 48 frame animation with 16 context_length (uniform)

![t2i_context_lat_ups_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/f0c736ee-d491-4c1d-9224-098576ca6cd0)

[aaa_readme_up_00009_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/f7a45f81-e700-4bfe-9fdd-fbcaa4fa8a4e)



### txt2img w/ latent upscale (full denoise on upscale)

![t2i_lat_ups_full_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/2d635a9c-634a-41d3-b358-837a6d956f19)

![aaa_readme_up_00010_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/804610de-18ec-43af-9af2-4a83cf31d16b)

[aaa_readme_up_00012_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/3eb575cf-92dd-434a-b3db-1a2064ff0033)



### txt2img w/ latent upscale (full denoise on upscale) - 48 frame animation with 16 context_length (uniform)

![t2i_context_lat_ups_full_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/9ee6ec0b-c5f8-4b21-a0af-9cd5c0b40061)

[aaa_readme_up_00014_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/034aff4c-f814-4b87-b5d1-407b1089af0d)



### txt2img w/ ControlNet-stabilized latent-upscale (partial denoise on upscale, Scaled Soft ControlNet Weights)

![t2i_lat_ups_softcontrol_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/149e18ff-319a-4e55-bdeb-7261cac0b510)

![aaa_readme_up_00017_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/221954cc-95df-4e0c-8ec9-266d0108dad4)

[aaa_readme_up_00019_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/b562251d-a4fb-4141-94dd-9f8bca9f3ce8)



### txt2img w/ ControlNet-stabilized latent-upscale (partial denoise on upscale, Scaled Soft ControlNet Weights) 48 frame animation with 16 context_length (uniform)

![t2i_context_lat_ups_softcontrol_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/229027ef-56c0-4b3a-8fe2-bbd4b60a70a6)

[aaa_readme_up_00003_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/0f57c949-0af3-4da4-b7c4-5c1fb1549927)



### txt2img w/ Initial ControlNet input (using Normal LineArt preprocessor on first txt2img as an example)

![t2i_initcn_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/f0895bf3-eeaa-4fae-a181-ffbd3ec8acf1)

![aaa_readme_cn_00001_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/ddf4b18e-904b-470f-9156-b65e9a16a694)

[aaa_readme_cn_00006_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/ef1603c8-636f-4f85-8122-bf53b553e263)


### txt2img w/ Initial ControlNet input (using Normal LineArt preprocessor on first txt2img 48 frame as an example) 48 frame animation with 16 context_length (uniform)

![t2i_context_initcn_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/26a8ada1-cd25-4413-b90e-89f92cd749ae)

![aaa_readme_cn_00009_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/09124505-3c41-46d7-abfa-0805390c23cb)

[aaa_readme_cn_00010_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/e32097fa-8b76-455d-9e52-cb55d0e1b357)


### txt2img w/ Initial ControlNet input (using OpenPose images) + latent upscale w/ full denoise

![t2i_openpose_upscale_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/2832e93b-5afb-42d5-9eca-32a353f41a21)


(open_pose images provided courtesy of toyxyz)

![AA_openpose_cn_gif_00001_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/23291941-864d-495a-8ba8-d02e05756396)

![aaa_readme_cn_00032_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/621a2ca6-2f08-4ed1-96ad-8e6635303173)

[aaa_readme_cn_00033_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/c5df09a5-8c64-4811-9ecf-57ac73d82377)



### txt2img w/ Initial ControlNet input (using OpenPose images) + latent upscale w/ full denoise, 48 frame animation with 16 context_length (uniform)

![t2i_context_openpose_upscale_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/d50bc565-fc4e-482c-9b51-ff77016b6712)

(open_pose images provided courtesy of toyxyz)

![aaa_readme_preview_00002_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/028a1e9e-37b5-477d-8665-0e8723306d65)

[aaa_readme_cn_00024_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/8f4c840c-06a2-4c64-b97e-568dd5ff6f46)



### img2img 

TODO: fill this out with a few useful ways, some using control net tile. I'm sorry there is nothing here right now, I have a lot of code to write. I'll try to fill this section out + Advance ControlNet use piece by piece.



## Known Issues

### Some motion models have visible watermark on resulting images (especially when using mm_sd_v15)

Training data used by the authors of the AnimateDiff paper contained Shutterstock watermarks. Since mm_sd_v15 was finetuned on finer, less drastic movement, the motion module attempts to replicate the transparency of that watermark and does not get blurred away like mm_sd_v14. Using other motion modules, or combinations of them using Advanced KSamplers should alleviate watermark issues.
