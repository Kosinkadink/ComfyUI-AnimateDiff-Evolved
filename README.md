# AnimateDiff for ComfyUI

[AnimateDiff](https://github.com/guoyww/AnimateDiff/) integration for ComfyUI, adapts from [sd-webui-animatediff](https://github.com/continue-revolution/sd-webui-animatediff). Please read the original repo README for more information.

## How to Use

1. Clone this repo into `custom_nodes` folder.
2. Download motion modules from [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI) | [HuggingFace](https://huggingface.co/guoyww/animatediff) | [CivitAI](https://civitai.com/models/108836) | [Baidu NetDisk](https://pan.baidu.com/s/18ZpcSM6poBqxWNHtnyMcxg?pwd=et8y). You only need to download one of `mm_sd_v14.ckpt` | `mm_sd_v15.ckpt`. Put the model weights under `comfyui-animatediff/models/`. DO NOT change model filename.

## Samples

### txt2img

<img width="1254" alt="ComfyUI AnimateDiff Usage" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/a88e2141-c55f-4bdb-b6ca-9155b6639114">

![AnimateDiff_00001](https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/e48f148a-886b-4a0d-b589-9fa795b06936)

### img2img
<img width="1121" alt="Screenshot 2023-07-22 at 22 08 00" src="https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/600f96b0-df21-4437-917f-7eda35ab6363">

![AnimateDiff_00002](https://github.com/ArtVentureX/comfyui-animatediff/assets/133728487/c78d64b9-b308-41ec-9804-bbde654d0b47)

## Known Issues

### GIF split into multiple scenes

![AnimateDiff_00007_](https://github.com/ArtVentureX/comfyui-animatediff/assets/8894763/e6cd53cb-9878-45da-a58a-a15851882386)

This is usually due to memory (VRAM) is not enough to process the whole image batch at the same time. Try reduce the image size and frame number.

### GIF has Wartermark after update to the latest version

See https://github.com/continue-revolution/sd-webui-animatediff/issues/31

As mentioned in the issue thread, it seems to be due to the training dataset. The new version is the correct implementation and produces smoother GIFs compared to the older version.

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
