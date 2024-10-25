
# Samples (download or drag images of the workflows into ComfyUI to instantly load the corresponding workflows!)

NOTE: I've scaled down the gifs to 0.75x size to make them take up less space on the README.
The updated workflows have included Context Options and Sample Settings connected. The Context Options (and FreeNoise) do nothing unless context windows are triggered.

### txt2vid

| Result                                                                                               |
| ---------------------------------------------------------------------------------------------------- |
| ![readme_00461](https://github.com/user-attachments/assets/e46e1a8b-cb50-4c6c-ad0e-07bfd75c6657)     |
| Workflow                                                                                             |
| ![workflow-txt2vid](https://github.com/user-attachments/assets/999f90a6-5958-4c7d-8dd6-4847f6de0d37) |

### txt2vid - (prompt travel)

| Result                                                                                                      |
| ----------------------------------------------------------------------------------------------------------- |
| ![readme_00463](https://github.com/user-attachments/assets/4c3e698c-2388-437a-b7a1-7857403a569a)            |
| Workflow                                                                                                    |
| ![workflow-txt2vid-travel](https://github.com/user-attachments/assets/c3ce95bb-b98a-40d6-bb9c-66dabf325eb7) |

### txt2vid - 32 frame animation with 16 context_length

| Result                                                                                                        |
| ------------------------------------------------------------------------------------------------------------- |
| ![readme_00475](https://github.com/user-attachments/assets/576d0293-1d32-4e9e-8ee8-124fc9421276)              |
| Workflow                                                                                                      |
| ![workflow-txt2vid-32frames](https://github.com/user-attachments/assets/0a320d9c-604b-4ac1-afe9-cc5c747f2118) |

### txt2vid - 32 frame animation with 16 context_length + ContextRef

Compared to without ContextRef, this tries to make the rest of the animation be more similar to the first context window.

| Result                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------ |
| ![readme_00474](https://github.com/user-attachments/assets/0870cea5-071c-42b1-acfb-4174bcb12d6f)                         |
| Workflow                                                                                                                 |
| ![workflow-txt2vid-32frames-contextref](https://github.com/user-attachments/assets/99ed4955-4a14-471b-9a53-d7791496de37) |


# Old Samples (TODO: update all of these + add new ones SOON)

### txt2img - 32 frame animation with 16 context_length (uniform) - PanLeft and ZoomOut Motion LoRAs

![t2i_context_mlora_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/41ec4141-389c-4ef4-ae3e-a963a0fa841f)

![aaa_readme_00094_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/14abee9a-5500-4d14-8632-15ac77ba5709)

[aaa_readme_00095_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/d730ae2e-188c-4a61-8a6d-bd48f60a2d07)


### txt2img w/ latent upscale (partial denoise on upscale)

![t2i_lat_ups_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/521991dd-8e39-4fed-9970-514507c75067)

![aaa_readme_up_00001_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/f4199e25-c839-41ed-8986-fb7dbbe2ac52)

[aaa_readme_up_00002_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/2f44342f-3fd8-4863-8e3d-360377d608b7)



### txt2img w/ latent upscale (partial denoise on upscale) - PanLeft and ZoomOut Motion LoRAs

![t2i_mlora_lat_ups_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/f34882de-7dd4-4264-8f59-e24da350be2a)

![aaa_readme_up_00023_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/e2ca5c0c-b5d9-42de-b877-4ed29db81eb9)

[aaa_readme_up_00024_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/414c16d8-231c-422f-8dfc-a93d4b68ffcc)



### txt2img w/ latent upscale (partial denoise on upscale) - 48 frame animation with 16 context_length (uniform)

![t2i_lat_ups_full_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/a1ebc14e-853e-4cda-9cda-9a7553fa3d85)

[aaa_readme_up_00009_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/f7a45f81-e700-4bfe-9fdd-fbcaa4fa8a4e)



### txt2img w/ latent upscale (full denoise on upscale)

![t2i_lat_ups_full_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/5058f201-3f52-4c48-ac7e-525c3c8f3df3)

![aaa_readme_up_00010_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/804610de-18ec-43af-9af2-4a83cf31d16b)

[aaa_readme_up_00012_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/3eb575cf-92dd-434a-b3db-1a2064ff0033)



### txt2img w/ latent upscale (full denoise on upscale) - 48 frame animation with 16 context_length (uniform)

![t2i_context_lat_ups_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/7b9ec22b-d4e0-4083-9846-5743ed90583e)

[aaa_readme_up_00014_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/034aff4c-f814-4b87-b5d1-407b1089af0d)



### txt2img w/ ControlNet-stabilized latent-upscale (partial denoise on upscale, Scaled Soft ControlNet Weights)

![t2i_lat_ups_softcontrol_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/c769c2bd-5aac-48d0-92b7-d73c422d4863)

![aaa_readme_up_00017_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/221954cc-95df-4e0c-8ec9-266d0108dad4)

[aaa_readme_up_00019_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/b562251d-a4fb-4141-94dd-9f8bca9f3ce8)



### txt2img w/ ControlNet-stabilized latent-upscale (partial denoise on upscale, Scaled Soft ControlNet Weights) 48 frame animation with 16 context_length (uniform)

![t2i_context_lat_ups_softcontrol_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/798567a8-4ef0-4814-aeeb-4f770df8d783)

[aaa_readme_up_00003_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/0f57c949-0af3-4da4-b7c4-5c1fb1549927)



### txt2img w/ Initial ControlNet input (using Normal LineArt preprocessor on first txt2img as an example)

![t2i_initcn_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/caa7abdf-7ba0-456c-9fa4-547944ea6e72)

![aaa_readme_cn_00002_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/055ef87c-50c6-4bb9-b35e-dd97916b47cc)

[aaa_readme_cn_00003_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/9c9d425d-2378-4af0-8464-2c6c0d1a68bf)



### txt2img w/ Initial ControlNet input (using Normal LineArt preprocessor on first txt2img 48 frame as an example) 48 frame animation with 16 context_length (uniform)

![t2i_context_initcn_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/f9de2711-dcfd-4fea-8b3b-31e3794fbff9)

![aaa_readme_cn_00005_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/6bf14361-5b09-4305-b2a7-f7babad4bd14)

[aaa_readme_cn_00006_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/5d3665b7-c2da-46a1-88d8-ab43ba8eb0c6)



### txt2img w/ Initial ControlNet input (using OpenPose images) + latent upscale w/ full denoise

![t2i_openpose_upscale_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/306a40c4-0591-496d-a320-c33f0fc4b3d2)

(open_pose images provided courtesy of toyxyz)

![AA_openpose_cn_gif_00001_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff/assets/7365912/23291941-864d-495a-8ba8-d02e05756396)

![aaa_readme_cn_00032_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/621a2ca6-2f08-4ed1-96ad-8e6635303173)

[aaa_readme_cn_00033_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/c5df09a5-8c64-4811-9ecf-57ac73d82377)



### txt2img w/ Initial ControlNet input (using OpenPose images) + latent upscale w/ full denoise, 48 frame animation with 16 context_length (uniform)

![t2i_context_openpose_upscale_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/a931af6f-bf6a-40d3-bd55-1d7bad32e665)

(open_pose images provided courtesy of toyxyz)

![aaa_readme_preview_00002_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/028a1e9e-37b5-477d-8665-0e8723306d65)

[aaa_readme_cn_00024_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/8f4c840c-06a2-4c64-b97e-568dd5ff6f46)



### img2img 

TODO: fill this out with a few useful ways, some using control net tile. I'm sorry there is nothing here right now, I have a lot of code to write. I'll try to fill this section out + Advance ControlNet use piece by piece.

