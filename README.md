# AnimateDiff for ComfyUI

Improved [AnimateDiff](https://github.com/guoyww/AnimateDiff/) integration for ComfyUI, initially adapted from [sd-webui-animatediff](https://github.com/continue-revolution/sd-webui-animatediff) but changed greatly since then. Please read the AnimateDiff repo README for more information about how it works at its core.

Examples shown here will also often make use of two helpful set of nodes:
- [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet) for loading files in batches and controlling which latents should be affected by the ControlNet inputs (work in progress, will include more advance workflows + features for AnimateDiff usage later).
- [comfy_controlnet_preprocessors](https://github.com/Fannovel16/comfy_controlnet_preprocessors) for ControlNet preprocessors not present in vanilla ComfyUI; this repo is archived, and future development by the dev will happen here: [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux). While most preprocessors are common between the two, some give different results. Workflows linked here use the archived version, comfy_controlnet_preprocessors.

## How to Use

1. Clone this repo into `custom_nodes` folder.
2. Download motion modules from [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI) | [HuggingFace](https://huggingface.co/guoyww/animatediff) | [CivitAI](https://civitai.com/models/108836) | [Baidu NetDisk](https://pan.baidu.com/s/18ZpcSM6poBqxWNHtnyMcxg?pwd=et8y). You can download one or more motion models. They can be renamed if you want. More motion modules are being trained by the community - if I am made aware of any good ones, I will link here as well. (TODO: create .safetensor versions of the motion modules and share them here.)
3. Get creative! If it works for normal image generation, it (probably) will work for AnimateDiff generations. Latent upscales? Go for it. ControlNets, one or more stacked? You betcha. Masking the conditioning of ControlNets to only affect part of the animation? Sure. Try stuff and you will be surprised by what you can do.

## Samples (still images of animation [not the workflow images] contains embeded workflow - download and drag it into ComfyUI to instantly load the workflow)

### txt2img

![txt2image_workflow](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/21d35c6d-1e61-4d05-81a1-87a07da35e67)

![AA_gif_00002_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/91933fb2-5b0b-4f41-a57a-ebebb604bd9d)

![AA_readme_00289_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/6ed622ef-cc79-42bc-81c7-81a69eb48f38)


### txt2img w/ latent upscale (partial denoise on upscale)

![txt2image_upscale_partialdenoise_workflow](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/03d8b53e-5f78-4d92-8875-0f36c18f41fb)

![AA_upscale_gif_00007_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/0cb2ca7e-8666-4abc-86f2-f24a20ff4bed)

![AA_upscale_readme_00097_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/82dfc0f8-e8a1-43ff-b8cf-a77875908180)


### txt2img w/ latent upscale (full denoise on upscale)

![txt2image_upscale_workflow](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/0eff4c23-ed61-412c-a9cf-533e26b4fe3a)

![AA_upscale_gif_00001_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/4ca8abd2-0b48-41d6-9eea-ed1467a68f5f)

![AA_upscale_readme_00001_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/328ba63a-67a7-4ce7-937e-97ceebc4d79f)


### txt2img w/ ControlNet-stabilized latent-upscale (partial denoise on upscale, Scaled Soft ControlNet Weights)

![txt2image_upscale_controlnetsoftweights_partialdenoise_workflow](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/0653c7be-c2b1-4999-bc47-a1573bc462fc)

![AA_upscale_gif_00009_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/4f03d03d-839b-484d-b612-8add086a6b8b)

![AA_upscale_readme_00129_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/41473028-0dc2-4bdb-b70d-c96cb25feede)


### txt2img w/ ControlNet-stabilized latent-upscale (full denoise on upscale)

![txt2image_upscale_controlnet_workflow](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/54167762-55ce-4776-a09e-e655c5fe049a)

![AA_upscale_controlnet_gif_00006_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/480c9bed-132c-489c-9682-39856b87fedb)

![AA_upscale_readme_00081_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/13476d34-2d46-43e6-af88-ae43e44cb6de)


### txt2img w/ Initial ControlNet input (using LineArt preprocessor on first txt2img as an example)

![txt2image_controlnet_workflow](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/4d48ee81-42e3-4a27-b9ff-81b662bc602b)

![AA_controlnet_gif_00017_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/de149c0f-bc1d-4bb9-8b4d-b10686e5b09f)

![AA_readme_00449_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/9261a7ab-4272-4799-90b3-ef5c774e650a)


### txt2img w/ Initial ControlNet input (using OpenPose images) + latent upscale w/ full denoise

![txt2image_openpose_controlnet_upscale_workflow](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/c8330b1f-7d70-42b8-8189-bada4d889c1f)

(open_pose images provided courtesy of toyxyz)
![AA_openpose_cn_gif_00001_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/23291941-864d-495a-8ba8-d02e05756396)

![AA_gif_00029_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/8367b24e-dfe5-4942-8e21-ac5a562be731)

![AA_readme_00657_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/af2b6012-a0eb-4802-8785-2e05832e0d1c)


### img2img (TODO: this is outdated and still shows the old flickering version, update this)
<img width="1121" alt="Screenshot 2023-07-22 at 22 08 00" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/600f96b0-df21-4437-917f-7eda35ab6363">

![AnimateDiff_00002](https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/c78d64b9-b308-41ec-9804-bbde654d0b47)


## Upcoming features (aka TODO):
- Nodes for saving videos, saving generated files into a timestamped folder instead of all over ComfyUI output dir.
- Moving-window latent implementation for generating arbitrarily-long animations instead of being capped at 24 frames (moving window will still be limited to up to 24 frames).
- Add examples of using ControlNet to ease one image/controlnet input into another, and also add more nodes to Advanced-ControlNet to make it easier to do so


## Known Issues

### Large resolutions may cause xformers to throw a CUDA error concerning a misconfigured value despite being within VRAM limitations.

Not sure why this happens, seems like an xformers bug. When encountered, the workaround is to boot ComfyUI with the "--disable-xformers" argument.

### Some of the experimental dropdowns in AnimateDiff Loader node may not properly reload after ComfyUI is restarted and page not being refreshed.

Currently figuring out why, but should not be a common occurance.

### GIF has Watermark (especially when using mm_sd_v15)

Training data used by the authors of the AnimateDiff paper contained Shutterstock watermarks. Since mm_sd_v15 was finetuned on finer, less drastic movement, the motion module attempts to replicate the transparency of that watermark and does not get blurred away like mm_sd_v14. Community finetunes of motion modules should eventually create equivalent (or better) results without the watermark. Until then, you'll need some good RNG or stick with mm_sd_v14, depending on your application.
