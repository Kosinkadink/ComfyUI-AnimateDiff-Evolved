# AnimateDiff for ComfyUI

Improved [AnimateDiff](https://github.com/guoyww/AnimateDiff/) integration for ComfyUI, as well as advanced sampling options dubbed Evolved Sampling usable outside of AnimateDiff. Please read the AnimateDiff repo README and Wiki for more information about how it works at its core.

AnimateDiff workflows will often make use of these helpful node packs:
- [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet) for making ControlNets work with Context Options and controlling which latents should be affected by the ControlNet inputs. Includes SparseCtrl support. Maintained by me.
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) for loading videos, combining images into videos, and doing various image/latent operations like appending, splitting, duplicating, selecting, or counting. Actively maintained by AustinMroz and I.
- [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) for ControlNet preprocessors not present in vanilla ComfyUI. Maintained by Fannovel16.
- [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) for IPAdapter support. Maintained by cubiq (matt3o).
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) for miscellaneous nodes including selecting coordinates for animated GLIGEN. Maintained by kijai.
- [ComfyUI_FizzNodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes) for an alternate way to do prompt-travel functionality with the BatchPromptSchedule node. Maintained by FizzleDorf.

# Installation

## If using ComfyUI Manager:

1. Look for ```AnimateDiff Evolved```, and be sure the author is ```Kosinkadink```. Install it.
![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/2c7f29e1-d024-49e1-9eb0-d38070142584)


## If installing manually:
1. Clone this repo into `custom_nodes` folder.

# Model Setup:
1. Download motion modules. You will need at least 1. Different modules produce different results.
   - Original models ```mm_sd_v14```, ```mm_sd_v15```, ```mm_sd_v15_v2```, ```v3_sd15_mm```: [HuggingFace](https://huggingface.co/guoyww/animatediff/tree/cd71ae134a27ec6008b968d6419952b0c0494cf2) | [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI) | [CivitAI](https://civitai.com/models/108836)
   - Stabilized finetunes of mm_sd_v14, ```mm-Stabilized_mid``` and ```mm-Stabilized_high```, by **manshoety**: [HuggingFace](https://huggingface.co/manshoety/AD_Stabilized_Motion/tree/main)
   - Finetunes of mm_sd_v15_v2, ```mm-p_0.5.pth``` and ```mm-p_0.75.pth```, by **manshoety**: [HuggingFace](https://huggingface.co/manshoety/beta_testing_models/tree/main)
   - Higher resolution finetune,```temporaldiff-v1-animatediff```  by **CiaraRowles**: [HuggingFace](https://huggingface.co/CiaraRowles/TemporalDiff/tree/main)
   - FP16/safetensor versions of vanilla motion models, hosted by **continue-revolution** (takes up less storage space, but uses up the same amount of VRAM as ComfyUI loads models in fp16 by default): [HuffingFace](https://huggingface.co/conrevo/AnimateDiff-A1111/tree/main)
2. Place models in one of these locations (you can rename models if you wish):
   - ```ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/models```
   - ```ComfyUI/models/animatediff_models```
3. Optionally, you can use Motion LoRAs to influence movement of v2-based motion models like mm_sd_v15_v2.
   - [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI?usp=sharing) | [HuggingFace](https://huggingface.co/guoyww/animatediff) | [CivitAI](https://civitai.com/models/108836/animatediff-motion-modules)
   - Place Motion LoRAs in one of these locations (you can rename Motion LoRAs if you wish): 
      -  ```ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora```
      -  ```ComfyUI/models/animatediff_motion_lora```
4. Get creative! If it works for normal image generation, it (probably) will work for AnimateDiff generations. Latent upscales? Go for it. ControlNets, one or more stacked? You betcha. Masking the conditioning of ControlNets to only affect part of the animation? Sure. Try stuff and you will be surprised by what you can do. Samples with workflows are included below.

NOTE: you can also use custom locations for models/motion loras by making use of the ComfyUI ```extra_model_paths.yaml``` file. The id for motion model folder is ```animatediff_models``` and the id for motion lora folder is ```animatediff_motion_lora```.


# Features 
- Compatible with almost any vanilla or custom KSampler node.
- ControlNet, SparseCtrl, and IPAdapter support
- Infinite animation length support via sliding context windows across whole unet (Context Options) and/or within motion module (View Options)
- Scheduling Context Options to change across different points in the sampling process
- FreeInit and FreeNoise support (FreeInit is under iteration opts, FreeNoise is in SampleSettings' noise_type dropdown)
- Mixable Motion LoRAs from [original AnimateDiff repository](https://github.com/guoyww/animatediff/) implemented. Caveat: the original loras really only work on v2-based motion models like ```mm_sd_v15_v2```, ```mm-p_0.5.pth```, and ```mm-p_0.75.pth```.
     - UPDATE: New motion LoRAs without the v2 limitation can now be trained via the [AnimateDiff-MotionDirector repo](https://github.com/ExponentialML/AnimateDiff-MotionDirector). Shoutout to ExponentialML for implementing MotionDirector for AnimateDiff purposes!
- Prompt travel using built-in Prompt Scheduling nodes, or BatchPromptSchedule node from [ComfyUI_FizzNodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes)
- Scale and Effect multival inputs to control motion amount and motion model influence on generation.
     - Can be float, list of floats, or masks
- Custom noise scheduling via Noise Types, Noise Layers, and seed_override/seed_offset/batch_offset in Sample Settings and related nodes
- AnimateDiff model v1/v2/v3 support
- Using multiple motion models at once via Gen2 nodes (each supporting 
- [HotshotXL](https://huggingface.co/hotshotco/Hotshot-XL/tree/main) support (an SDXL motion module arch), ```hsxl_temporal_layers.safetensors```.
     - NOTE: You will need to use ```autoselect``` or ```linear (HotshotXL/default)``` beta_schedule, the sweetspot for context_length or total frames (when not using context) is 8 frames, and you will need to use an SDXL checkpoint.
- [AnimateDiff-SDXL](https://github.com/guoyww/AnimateDiff/) support, with corresponding model. Still in beta after several months.
     - NOTE: You will need to use ```autoselect``` or ```linear (AnimateDiff-SDXL)``` beta_schedule. Other than that, same rules of thumb apply to AnimateDiff-SDXL as AnimateDiff.
- [AnimateLCM](https://huggingface.co/wangfuyun/AnimateLCM) support
     - NOTE: You will need to use ```autoselect``` or ```lcm``` or ```lcm[100_ots]``` beta_schedule. To use fully with LCM, be sure to use appropriate LCM lora, use the ```lcm``` sampler_name in KSampler nodes, and lower cfg to somewhere around 1.0 to 2.0. Don't forget to decrease steps (minimum = ~4 steps), since LCM converges faster (less steps). Increase step count to increase detail as desired.
- [AnimateLCM-I2V](https://huggingface.co/wangfuyun/AnimateLCM-I2V) support, big thanks to [Fu-Yun Wang](https://github.com/G-U-N) for providing me the original diffusers code he created during his work on the paper
     - NOTE: Requires same settings as described for AnimateLCM above. Requires ```Apply AnimateLCM-I2V Model``` Gen2 node usage so that ```ref_latent``` can be provided; use ```Scale Ref Image and VAE Encode``` node to preprocess input images. While this was intended as an img2video model, I found it works best for vid2vid purposes with ```ref_drift=0.0```, and to use it for only at least 1 step before switching over to other models via chaining with toher Apply AnimateDiff Model (Adv.) nodes. The ```apply_ref_when_disabled``` can be set to True to allow the img_encoder to do its thing even when the ```end_percent``` is reached. AnimateLCM-I2V is also extremely useful for maintaining coherence at higher resolutions (with ControlNet and SD LoRAs active, I could easily upscale from 512x512 source to 1024x1024 in a single pass). TODO: add examples
- [CameraCtrl](https://github.com/hehao13/CameraCtrl) support, with the pruned model you must use here: [CameraCtrl_pruned.safetensors](https://huggingface.co/Kosinkadink/CameraCtrl/tree/main)
     - NOTE: Requires AnimateDiff SD1.5 models, and was specifically trained for v3 model. Gen2 only, with helper nodes provided under Gen2/CameraCtrl submenu.
- [PIA](https://github.com/open-mmlab/PIA) support, with the model [pia.ckpt](https://huggingface.co/Leoxing/PIA/tree/main)
     - NOTE: You will need to use ```autoselect``` or ```sqrt_linear (AnimateDiff)``` beta_schedule. Requires ```Apply AnimateDiff-PIA Model``` Gen2 node usage if you want to actually provide input images. The ```pia_input``` can be provided via the paper's presets (```PIA Input [Paper Presets]```) or by manually entering values (```PIA Input [Multival]```).
- AnimateDiff Keyframes to change Scale and Effect at different points in the sampling process.
- fp8 support; requires newest ComfyUI and torch >= 2.1 (decreases VRAM usage, but changes outputs)
- Mac M1/M2/M3 support
- Usage of Context Options and Sample Settings outside of AnimateDiff via Gen2 Use Evolved Sampling node
- Maskable and Schedulable SD LoRA (and Models as LoRA) for both AnimateDiff and StableDiffusion usage via LoRA Hooks
- Per-frame GLIGEN coordinates control
     - Currently requires GLIGENTextBoxApplyBatch from KJNodes to do so, but I will add native nodes to do this soon.
- Image Injection mid-sampling
- ContextRef and NaiveReuse (novel cross-context consistency techniques)

## Upcoming Features
- Example workflows for **every feature** in AnimateDiff-Evolved repo, nodes will have usage descriptions (currently Value/Prompt Scheduling nodes have them), and YouTube tutorials/documentation
- [UniCtrl](https://github.com/XuweiyiChen/UniCtrl) support
- Unet-Ref support so that a bunch of papers can be ported over
- [StoryDiffusion](https://github.com/HVision-NKU/StoryDiffusion) implementation
- Merging motion model weights/components, including per block customization
- Maskable Motion LoRA
- Timestep schedulable GLIGEN coordinates
- Dynamic memory management for motion models that load/unload at different start/end_percents
- Anything else AnimateDiff-related that comes out

# [Basic Usage and Nodes](./documentation/nodes)

# [Samples](./documentation/samples)



## Known Issues

### Some motion models have visible watermark on resulting images (especially when using mm_sd_v15)

Training data used by the authors of the AnimateDiff paper contained Shutterstock watermarks. Since mm_sd_v15 was finetuned on finer, less drastic movement, the motion module attempts to replicate the transparency of that watermark and does not get blurred away like mm_sd_v14. Using other motion modules, or combinations of them using Advanced KSamplers should alleviate watermark issues.
