# AnimateDiff for ComfyUI

Improved [AnimateDiff](https://github.com/guoyww/AnimateDiff/) integration for ComfyUI, initially adapted from [sd-webui-animatediff](https://github.com/continue-revolution/sd-webui-animatediff) but changed greatly since then. Please read the AnimateDiff repo README for more information about how it works at its core.

Examples shown here will also often make use of two helpful set of nodes:
- [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet) for loading files in batches and controlling which latents should be affected by the ControlNet inputs (work in progress, will include more advance workflows + features for AnimateDiff usage later).
- [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) for ControlNet preprocessors not present in vanilla ComfyUI.

## How to Use

1. Clone this repo into `custom_nodes` folder.
2. Download motion modules from [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI) | [HuggingFace](https://huggingface.co/guoyww/animatediff) | [CivitAI](https://civitai.com/models/108836) | [Baidu NetDisk](https://pan.baidu.com/s/18ZpcSM6poBqxWNHtnyMcxg?pwd=et8y). You can download one or more motion models. They can be renamed if you want. More motion modules are being trained by the community - if I am made aware of any good ones, I will link here as well. (TODO: create .safetensor versions of the motion modules and share them here.)

## Samples

### txt2img

<img width="1254" alt="ComfyUI AnimateDiff Usage" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/a88e2141-c55f-4bdb-b6ca-9155b6639114">

![AnimateDiff_00001](https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/e48f148a-886b-4a0d-b589-9fa795b06936)

### img2img
<img width="1121" alt="Screenshot 2023-07-22 at 22 08 00" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/600f96b0-df21-4437-917f-7eda35ab6363">

### ControlNet full animation workflow

### ControlNet keyframes workflow (in progress, will add more helpful nodes to Advanced-ControlNet soon and change that implementation)

![AnimateDiff_00002](https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/c78d64b9-b308-41ec-9804-bbde654d0b47)

## Upcoming features (aka TODO):
- Nodes for saving videos, saving generated files into a timestamped folder instead of all over ComfyUI output dir.
- Moving-window latent implementation for generating arbitrarily-long animations instead of being capped at 24 frames (moving window will still be limited to up to 24 frames).


## Known Issues

### Large resolutions may cause xformers to throw a CUDA error concerning a misconfigured value despite being within VRAM limitations.

Not sure why this happens, seems like an xformers bug. When encountered, the workaround is to boot ComfyUI with the "--disable-xformers" argument.

### Some of the experimental dropdowns in AnimateDiff Loader node may not properly reload after ComfyUI is restarted and page not being refreshed.

Currently figuring out why, but should not be a common occurance.

### GIF has Watermark (especially when using mm_sd_v15)

Training data used by the authors of the AnimateDiff paper contained Shutterstock watermarks. Since mm_sd_v15 was finetuned on finer, less drastic movement, the motion module attempts to replicate the transparency of that watermark and does not get blurred away like mm_sd_v14. Community finetunes of motion modules should eventually create equivalent (or better) results without the watermark. Until then, you'll need some good RNG or stick with mm_sd_v15, depending on your application.

<table  class="center">
    <tr>
    <td>Old revision</td>
    <td>New revision</td>
    </tr>
    <tr>
    <td><img src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/8f1a6233-875f-4f0c-aa60-ba93e73b7d64" /></td>
    <td><img src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/a2029eba-f519-437c-a0b5-1f881e099a20" /></td>
    </tr>
    <tr>
    <td><img src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/41ec449f-1955-466c-bd38-6f2a55d654f8" /></td>
    <td><img src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/766c2891-5d27-4052-99f9-be9862620919" /></td>
    </tr>
</table>


I played around with both version and found that the watermark only present in some models, not always. So I've brought back the old method and also created a new node with the new method. You can try both to find the best fit for each model.

![Screenshot 2023-07-28 at 18 14 14](https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/25cf6092-3e67-435e-86cc-43614ca7d6aa)
