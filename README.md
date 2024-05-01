# AnimateDiff for ComfyUI

Improved [AnimateDiff](https://github.com/guoyww/AnimateDiff/) integration for ComfyUI, as well as advanced sampling options dubbed Evolved Sampling usable outside of AnimateDiff. Please read the AnimateDiff repo README and Wiki for more information about how it works at its core.

AnimateDiff workflows will often make use of these helpful node packs:
- [ComfyUI_FizzNodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes) for prompt-travel functionality with the BatchPromptSchedule node. Maintained by FizzleDorf.
- [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet) for making ControlNets work with Context Options and controlling which latents should be affected by the ControlNet inputs. Includes SparseCtrl support. Maintained by me.
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) for loading videos, combining images into videos, and doing various image/latent operations like appending, splitting, duplicating, selecting, or counting. Actively maintained by AustinMroz and I.
- [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) for ControlNet preprocessors not present in vanilla ComfyUI. Maintained by Fannovel16.
- [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) for IPAdapter support. Maintained by cubiq (matt3o).
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) for miscellaneous nodes including selecting coordinates for animated GLIGEN. Maintained by kijai.

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
- Prompt travel using BatchPromptSchedule node from [ComfyUI_FizzNodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes)
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
- AnimateDiff Keyframes to change Scale and Effect at different points in the sampling process.
- fp8 support; requires newest ComfyUI and torch >= 2.1 (decreases VRAM usage, but changes outputs)
- Mac M1/M2/M3 support
- Usage of Context Options and Sample Settings outside of AnimateDiff via Gen2 Use Evolved Sampling node
- Maskable and Schedulable SD LoRA (and Models as LoRA) for both AnimateDiff and StableDiffusion usage via LoRA Hooks
- Per-frame GLIGEN coordinates control
     - Currently requires GLIGENTextBoxApplyBatch from KJNodes to do so, but I will add native nodes to do this soon.

## Upcoming Features
- Example workflows for **every feature** in AnimateDiff-Evolved repo, and hopefully a long Youtube video showing all features (Goal: mid-May)
- Maskable Motion LoRA (Goal: end of May/beginning of June)
- Timestep schedulable GLIGEN coordinates
- Dynamic memory management for motion models that load/unload at different start/end_percents
- [PIA](https://github.com/open-mmlab/PIA) support
- [UniCtrl](https://github.com/XuweiyiChen/UniCtrl) support
- Built-in prompt travel implementation
- Anything else AnimateDiff-related that comes out


# Basic Usage And Nodes

There are two families of nodes that can be used to use AnimateDiff/Evolved Sampling - **Gen1** and **Gen2**. Other than nodes marked specifically for Gen1/Gen2, all other nodes can be used for both Gen1 and Gen2.

Gen1 and Gen2 produce the exact same results (the backend code is identical), the only difference is in how the modes are used. Overall, Gen1 is the simplest way to use basic AnimateDiff features, while Gen2 separates model loading and application from the Evolved Sampling features. This means in practice, Gen2's Use Evolved Sampling node can be used without a model model, letting Context Options and Sample Settings be used without AnimateDiff.

In the following documentation, inputs/outputs will be color coded as follows:
- 🟩 - required inputs
- 🟨 - optional inputs
- 🟦 - start as widgets, can be converted to inputs
- 🟪 - output

## Gen1/Gen2 Nodes

| ① Gen1 ① | ② Gen2 ② |
|---|---|
| - All-in-One node<br/> - If same model is loaded by multiple Gen1 nodes, duplicates RAM usage. | - Separates model loading from application and Evolved Sampling<br/> - Enables no motion model usage while preserving Evolved Sampling features<br/> - Enables multiple motion model usage with Apply AnimateDiff Model (Adv.) Node|
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/a94029fd-5e74-467b-853c-c3ec4cf8a321)| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/8c050151-6cfb-4350-932d-a105af78a1ec)|
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/c7ae9ef3-b5cd-4800-b249-da2cb73c4c1e)| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/cffa21f7-0e33-45d1-9950-ad22eb229134) |


### Inputs
- 🟩*model*: StableDiffusion (SD) Model input.
- 🟦*model_name*: AnimateDiff (AD) model to load and/or apply during the sampling process. Certain motion models work with SD1.5, while others work with SDXL.
- 🟦*beta_schedule*: Applies selected beta_schedule to SD model; ```autoselect``` will automatically select the recommended beta_schedule for selected motion models - or will use_existing if no motion model selected for Gen2.
- 🟨*context_options*: Context Options node from the context_opts submenu - should be used when needing to go back the sweetspot of an AnimateDiff model. Works with no motion models as well (Gen2 only).
- 🟨*sample_settings*: Sample Settings node input - used to apply custom sampling options such as FreeNoise (noise_type), FreeInit (iter_opts), custom seeds, Noise Layers, etc. Works with no motion models as well (Gen2 only).
- 🟨*motion_lora*: For v2-based models, Motion LoRA will influence the generated movement. Only a few official motion LoRAs were released - soon, I will be working with some community members to create training code to create (and test) new Motion LoRAs that might work with non-v2 models.
- 🟨*ad_settings*: Modifies motion models during loading process, allowing the Positional Encoders (PEs) to be adjusted to extend a model's sweetspot or modify overall motion.
- 🟨*ad_keyframes*: Allows scheduling of ```scale_multival``` and ```effect_multival``` inputs across sampling timesteps.
- 🟨*scale_multival*: Uses a ```Multival``` input (defaults to ```1.0```). Previously called motion_scale, it directly influences the amount of motion generated by the model. With the Multival nodes, it can accept a float, list of floats, and/or mask inputs, allowing different scale to be applied to not only different frames, but different areas of frames (including per-frame).
- 🟨*effect_multival*: Uses a ```Multival``` input (defaults to ```1.0```). Determines the influence of the motion models on the sampling process. Value of ```0.0``` is equivalent to normal SD output with no AnimateDiff influence. With the Multival nodes, it can accept a float, list of floats, and/or mask inputs, allowing different effect amount to be applied to not only different frames, but different areas of frames (including per-frame).

#### Gen2-Only Inputs
- 🟨*motion_model*: Input for loaded motion_model.
- 🟨*m_models*: One (or more) motion models outputted from Apply AnimateDiff Model nodes.

#### Gen2 Adv.-Only Inputs
- 🟨*prev_m_models*: Previous applied motion models to use alongside this one.
- 🟨*start_percent*: Determines when connected motion_model should take effect (supercedes any ad_keyframes).
- 🟨*end_percent*: Determines when connected motion_model should stop taking effect (supercedes any ad_keyframes).

#### Gen1 (Legacy) Inputs
- 🟦*motion_scale*: legacy version of ```scale_multival```, can only be a float.
- 🟦*apply_v2_models_properly*: backwards compatible toggle for months-old workflows that used code that did not turn off groupnorm hack for v2 models. **Only affects v2 models, nothing else.** All nodes default this value to ```True``` now.

### Outputs
- 🟪*MODEL*: Injected SD model with Evolved Sampling/AnimateDiff.

#### Gen2-Only Outputs
- 🟪*MOTION_MODEL*: Loaded motion model.
- 🟪*M_MODELS*: One (or more) applied motion models, to be either plugged into Use Evolved Sampling or another Apply AnimateDiff Model (Adv.) node.


## Multival Nodes

For Multival inputs, these nodes allow the use of floats, list of floats, and/or masks to use as input. Scaled Mask node allows customization of dark/light areas of masks in terms of what the values correspond to.

| Node | Inputs |
|---|---|
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/d4c6a63f-703a-402b-989e-ab4d04141c7a) | 🟨*mask_optional*: Mask for float values - black means 0.0, white means 1.0 (multiplied by float_val). <br/> 🟦*float_val*: Float multiplier.|
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/bc100bec-0407-47c8-aebd-f74f2417711e) | 🟩*mask*: Mask for float values. <br/> 🟦*min_float_val*: Minimum value. <br/>🟦*max_float_val*: Maximum value. <br/> 🟦*scaling*: When ```absolute```, black means min_float_val, white means max_float_val. When ```relative```, darkest area in masks (total) means min_float_val, lighest area in massk (total) means max_float_val. |


## AnimateDiff Keyframe

Allows scheduling (in terms of timesteps) for scale_multival and effect_multival.

The two settings to determine schedule are ***start_percent*** and ***guarantee_steps***. When multiple keyframes have the same start_percent, they will be executed in the order they are connected, and run for guarantee_steps before moving on to the next node.

| Node |
|---|
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/dca73cdc-157a-47db-bed2-6ba584dceccd) |

### Inputs
- 🟨*prev_ad_keyframes*: Chained keyframes to create schedule.
- 🟨*scale_multival*: Value of scale to use for this keyframe.
- 🟨*effect_multival*: Value of effect to use for this keyframe.
- 🟦*start_percent*: Percent of timesteps to start usage of this keyframe. If multiple keyframes have same start_percent, order of execution is determined by their chained order, and will last for guarantee_steps timesteps.
- 🟦*guarantee_steps*: Minimum amount of steps the keyframe will be used - when set to 0, this keyframe will only be used when no other keyframes are better matches for current timestep.
- 🟦*inherit_missing*: When set to ```True```, any missing scale_multival or effect_multival inputs will inherit the previous keyframe's values - if the previous keyframe also inherits missing, the last inherited value will be used.


## Context Options and View Options

These nodes provide techniques used to extend the lengths of animations to get around the sweetspot limitations of AnimateDiff models (typically 16 frames) and HotshotXL model (8 frames). 

Context Options works by diffusing portions of the animation at a time, including main SD diffusion, ControlNets, IPAdapters, etc., effectively limiting VRAM usage to be equivalent to be context_length latents.

View Options, in contrast, work by portioning the latents seen by the motion model. This does NOT decrease VRAM usage, but in general is more stable and faster than Context Options, since the latents don't have to go through the whole SD unet.

Context Options and View Options can be combined to get the best of both worlds - longer context_length can be used to gain more stable output, at the cost of using more VRAM (since context_length determines how much SD sampling is done at the same time on the GPU). Provided you have the VRAM, you could also use Views Only Context Options to use only View Options (and automatically make context_length equivalent to full latents) to get a speed boost in return for the higher VRAM usage.

There are two types of Context/View Options: ***Standard*** and ***Looped***. ***Standard*** options do not cause looping in the output. ***Looped*** options, as the name implies, causes looping in the output (from end to beginning). Prior to the code rework, the only context available was the looping kind.

***I recommend using Standard Static at first when not wanting looped outputs.***

In the below animations, ***green*** shows the Contexts, and ***red*** shows the Views. TL;DR green is the amount of latents that are loaded into VRAM (and sampled), while red is the amount of latents that get passed into the motion model at a time.

### Context Options◆Standard Static
| Behavior |
|---|
| ![anim__00005](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/b26792d6-0f41-4f07-93aa-e5ee83f4d90e) <br/> (latent count: 64, context_length: 16, context_overlap: 4, total steps: 20)|

| Node | Inputs |
|---|---|
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/a4a5f38e-3a1b-4328-9537-ad17567aed75) | 🟦*context_length*: Amount of latents to diffuse at once.<br/> 🟦*context_overlap*: Minimum common latents between adjacent windows.<br/> 🟦*fuse_method*: Method for averaging results of windows.<br/> 🟦*use_on_equal_length*: When True, allows context to be used when latent count matches context_length.<br/> 🟦*start_percent*: When multiple Context Options are chained, allows scheduling.<br/> 🟦*guarantee_steps*: When scheduling contexts, determines the *minimum* amount of sampling steps context should be used.<br/> 🟦*context_length*: Amount of latents to diffuse at once.<br/> 🟨*prev_context*: Allows chaining of contexts.<br/> 🟨*view_options*: When context_length > view_length (unless otherwise specified), allows view_options to be used within each context window.|

### Context Options◆Standard Uniform
| Behavior |
|---|
| ![anim__00006](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/69707e3d-f49e-4368-89d5-616af2631594) <br/> (latent count: 64, context_length: 16, context_overlap: 4, context_stride: 1, total steps: 20) |
| ![anim__00010](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/7fc083b4-406f-4809-94ca-b389784adcab) <br/> (latent count: 64, context_length: 16, context_overlap: 4, context_stride: 2, total steps: 20) |

| Node | Inputs |
|---|---|
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/c2c8c7ea-66b6-408d-be46-1d805ecd64d1) | 🟦*context_length*: Amount of latents to diffuse at once.<br/> 🟦*context_overlap*: Minimum common latents between adjacent windows.<br/> 🟦*context_stride*: Maximum 2^(stride-1) distance between adjacent latents.<br/> 🟦*fuse_method*: Method for averaging results of windows.<br/> 🟦*use_on_equal_length*: When True, allows context to be used when latent count matches context_length.<br/> 🟦*start_percent*: When multiple Context Options are chained, allows scheduling.<br/> 🟦*guarantee_steps*: When scheduling contexts, determines the *minimum* amount of sampling steps context should be used.<br/> 🟦*context_length*: Amount of latents to diffuse at once.<br/> 🟨*prev_context*: Allows chaining of contexts.<br/> 🟨*view_options*: When context_length > view_length (unless otherwise specified), allows view_options to be used within each context window.|

### Context Options◆Looped Uniform
| Behavior |
|---|
| ![anim__00008](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/d08ac1c9-2cec-4c9e-b257-0a804448d41b) <br/> (latent count: 64, context_length: 16, context_overlap: 4, context_stride: 1, closed_loop: False, total steps: 20) |
| ![anim__00009](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/61e0311b-b623-423f-bbcb-eb4eb02e9002) <br/> (latent count: 64, context_length: 16, context_overlap: 4, context_stride: 1, closed_loop: True, total steps: 20) |

| Node | Inputs |
|---|---|
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/c2c8c7ea-66b6-408d-be46-1d805ecd64d1) | 🟦*context_length*: Amount of latents to diffuse at once.<br/> 🟦*context_overlap*: Minimum common latents between adjacent windows.<br/> 🟦*context_stride*: Maximum 2^(stride-1) distance between adjacent latents.<br/> 🟦*closed_loop*: When True, adds additional windows to enhance looping.<br/> 🟦*fuse_method*: Method for averaging results of windows.<br/> 🟦*use_on_equal_length*: When True, allows context to be used when latent count matches context_length - allows loops to be made when latent count == context_length.<br/> 🟦*start_percent*: When multiple Context Options are chained, allows scheduling.<br/> 🟦*guarantee_steps*: When scheduling contexts, determines the *minimum* amount of sampling steps context should be used.<br/> 🟦*context_length*: Amount of latents to diffuse at once.<br/> 🟨*prev_context*: Allows chaining of contexts.<br/> 🟨*view_options*: When context_length > view_length (unless otherwise specified), allows view_options to be used within each context window.|

### Context Options◆Views Only [VRAM⇈]
| Behavior |
|---|
| ![anim__00011](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/f2e422a4-c894-4e89-8f35-1964b89f369d) <br/> (latent count: 64, view_length: 16, view_overlap: 4, View Options◆Standard Static, total steps: 20) |

| Node | Inputs |
|---|---|
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/8cd6a0a4-ee8a-46c3-b04b-a100f87025b3) | 🟩*view_opts_req*: View_options to be used across all latents. <br/> 🟨*prev_context*: Allows chaining of contexts.<br/> |


There are View Options equivalent of these schedules:

### View Options◆Standard Static
| Behavior |
|---|
| ![anim__00012](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/7aee4ccb-b669-42fd-a1b5-2005003d5f8d) <br/> (latent count: 64, view_length: 16, view_overlap: 4, Context Options◆Standard Static, context_length: 32, context_overlap: 8, total steps: 20) |

| Node | Inputs |
|---|---|
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/4b22c73f-99cb-4781-bd33-e1b3db848207) | 🟦*view_length*: Amount of latents in context to pass into motion model at a time.<br/> 🟦*view_overlap*: Minimum common latents between adjacent windows.<br/> 🟦*fuse_method*: Method for averaging results of windows.<br/> |

### View Options◆Standard Uniform
| Behavior |
|---|
| ![anim__00015](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/faa2cd26-9f94-4fce-90b2-8acec84b444e ) <br/> (latent count: 64, view_length: 16, view_overlap: 4, view_stride: 1, Context Options◆Standard Static, context_length: 32, context_overlap: 8, total steps: 20) |

| Node | Inputs |
|---|---|
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/bbf017e6-3545-4043-ba41-fcbe2f54496a) | 🟦*view_length*: Amount of latents in context to pass into motion model at a time.<br/> 🟦*view_overlap*: Minimum common latents between adjacent windows.<br/> 🟦*view_stride*: Maximum 2^(stride-1) distance between adjacent latents.<br/> 🟦*fuse_method*: Method for averaging results of windows.<br/> |

### View Options◆Looped Uniform
| Behavior |
|---|
| ![anim__00016](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/8922b44b-cb19-4b2a-8486-2df8a46bf573) <br/> (latent count: 64, view_length: 16, view_overlap: 4, view_stride: 1, closed_loop: False, Context Options◆Standard Static, context_length: 32, context_overlap: 8, total steps: 20) |
| NOTE: this one is probably not going to come out looking well unless you are using this for a very specific reason. |

| Node | Inputs |
|---|---|
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/c58fe4d4-81a8-436b-8028-9e81c2ace18a) | 🟦*view_length*: Amount of latents in context to pass into motion model at a time.<br/> 🟦*view_overlap*: Minimum common latents between adjacent windows.<br/> 🟦*view_stride*: Maximum 2^(stride-1) distance between adjacent latents.<br/> 🟦*closed_loop*: When True, adds additional windows to enhance looping.<br/> 🟦*use_on_equal_length*: When True, allows context to be used when latent count matches context_length - allows loops to be made when latent count == context_length.<br/> 🟦*fuse_method*: Method for averaging results of windows.<br/> |

## Sample Settings

The Sample Settings node allows customization of the sampling process beyond what is exposed on most KSampler nodes. With its default values, it will NOT have any effect, and can safely be attached without changing any behavior.

TL;DR To use FreeNoise, select ```FreeNoise``` from the noise_type dropdown. FreeNoise does not decrease performance in any way. To use FreeInit, attach the FreeInit Iteration Options to the iteration_opts input. NOTE: FreeInit, despite it's name, works by resampling the latents ```iterations``` amount of times - this means if you use iteration=2, total sampling time will be exactly twice as slow since it will be performing the sampling twice.

Noise Layers with the inputs of the same name (or very close to same name) have same intended behavior as the ones for Sample Settings - refer to the inputs below.

| Node |
|---|
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/563a13cf-7aed-4acc-9ce3-1556660a34c2) |

### Inputs
- 🟨*noise_layers*: Customizable, stackable noise to add to/modify initial noise.
- 🟨*iteration_opts*: Options for determining if (and how) sampling should be repeated consecutively; if you want to check out FreeInit, this is how to use it.
- 🟨*seed_override*: Accepts a single int to use a seed instead of the seed passed into the KSampler, or a list of ints (like via FizzNodes' BatchedValueSchedule) to assign individual seeds to each latent in the batch.
- 🟦*seed_offset*: When not set to 0, adds value to current seed, predictably changing it, whatever the original seed may have been.
- 🟦*batch_offset*: When not set to 0, will 'offset' the noise as if the first latent was actually the batch_offset-nth latent, shifting all the noises over.
- 🟦*noise_type*: Selects type of noise to be generated. Values include:
   - **default**: generates different noise for all latents as usual.
   - **constant**: generates exact same noise for all latents (based on seed).
   - **empty**: generates no noise for all latents (as if noise was turned off).
   - **repeated_context**: repeats noise every context_length (or view_length) amount of latents; stabilizes longer generations, but has very obvious repetition.
   - **FreeNoise**: repeats noise such that it is repeated every context_length (or view_length), but the overlapped noise between contexts/views is shuffled to make repetition less prevelant while still achieving stabilization.
- 🟦*seed_gen*: Allows choosing between ComfyUI and Auto1111 methods of noise generation. One is not better than the other (noise distributions are the same), they are just different methods.
   - **comfy**: Noise is generated for the entire latent batch tensor at once based on the provided seed.
   - **auto1111**: Noise is generated individually for each latent, with each latent receiving an increasing +1 seed offset (first latent uses seed, second latent uses seed+1, etc.).
- 🟦*adapt_denoise_steps*: When True, KSamplers with a 'denoise' input will automatically scale down the total steps to run like the default options in Auto1111.
   - **True**: Steps will decrease with lower denoise, i.e. 20 steps with 0.5 denoise will be 10 total steps executed, but sigmas will be selected that still achieve 0.5 denoise. Trades speed for quality (since less steps are sampled).
   - **False**: Default behavior; 20 steps with 0.5 denoise will execute 20 steps.


## Iteration Options

These options allow KSamplers to re-sample the same latents without needing to chain multiple KSamplers together, and also allows specialized iteration behavior to implement features such as FreeInit.

### Default Iteration Options

Simply re-runs the KSampler, plugging in the output of the previous iteration into the next one. At the dafault iterations=1, it is no different than not having this node plugged in at all.

| Node | Inputs |
|---|---|
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/23c5e698-6eff-43cc-92e9-488e9b5ca96a) | 🟦*iterations*: Total amount of times KSampler should run back-to-back. <br/> 🟦*iter_batch_offset*: batch_offset to apply on each subsequent iteration. <br/> 🟦*iter_seed_offset*: seed_offset to apply on each subsequent iteration. |

### FreeInit Iteration Options

Implements [FreeInit](https://github.com/TianxingWu/FreeInit), which is the idea that AnimateDiff was trained on latents of existing videos (images with temporal coherence between them) that were then noised rather than from random initial noise, and that when noising existing latents, low-frequency data still remains in the noised latents. It combines the low-frequency noise from existing videos (or, as is the default behavior, the previous iteration) with the high-frequency noise in randomly generated noise to run the subsequent iterations. ***Each iteration is a full sample - 2 iterations means it will take twice as long to run as compared to having 1 iteration/no iteration_opts connected.***

When apply_to_1st_iter is False, the noising/low-freq/high-freq combination will not occur on the first iteration, with the assumption that there are no useful latents passed in to do the noise combining in the first place, thus requiring at least 2 iterations for FreeInit to take effect.

If you have an existing set of latents to use to get low-freq noise from, you may set apply_to_1st_iter to True, and then even if you set iterations=1, FreeInit will still take effect.

| Node |
|---|
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/21404e4f-ab67-44ed-8bf9-e510bc2571de) |

#### Inputs
- 🟦*iterations*: Total amount of times KSampler should run back-to-back. Refer to explanation above why it is 2 by default (and when it can be set to 1 instead).
- 🟦*init_type*: Code implementation for applying FreeInit.
   - ***FreeInit [sampler sigma]***: likely closest to intended implementation, and gets the sigma for noising from the sampler instead of the model (when possible).
   - ***FreeInit [model sigma]***: gets sigma for noising from the model; when using Custom KSampler, this is the method that will be used for both FreeInit options.
   - ***DinkInit_v1***: my initial, flawed implementation of FreeInit before I figured out how to exactly copy the noising behavior. By sheer luck and trial and error, I managed to have it actually sort of work with this method. Mainly for backwards compatibility now, but might produce useful results too.

- 🟦*apply_to_1st_iter*: When set to True, will do FreeInit low-freq/high-freq combo work even on the 1st iteration it runs Refer to explanation in the above FreeInit Iteration Options section for when this can be set to True.
- 🟦*init_type*: Code implementation for applying FreeInit.
- 🟦*iter_batch_offset*: batch_offset to apply on each subsequent iteration.
- 🟦*iter_seed_offset*: seed_offset to apply on each subsequent iteration. Defaults to 1 so that new random noise is used for each iteration.

- 🟦*filter*: Determines low-freq filter to apply to noise. Very technical, look into code/online resources to figure out how the individual filters act.
- 🟦*d_s*: Spatial parameter of filter (within latents, I think); very technical. Look into code/online resources if you wish to know what exactly it does.
- 🟦*d_t*: Temporal parameter of filter (across latents, I think); very technical. Look into code/online resources if you wish to know what exactly it does.
- 🟦*n_butterworth*: Only applies to ```butterworth``` filter; very technical. Look into code/online resources if you wish to know what exactly it does.
- 🟦*sigma_step*: Noising step to use/emulate when noising latents to then get low-freq noise out of. 999 actually means last (-1), and any number under 999 will mean the distance away from last. Leave at 999 unless you know what you're trying to do with it.


## Noise Layers

These nodes allow initial noise to be added onto, weighted, or replaced. In near future, I will add the ability for masks to 'move' the noise relative to the masks' movement instead of just 'cutting and pasting' the noise.

The inputs that are shared with Sample Settings have the same exact effect - only new option is in seed_gen_override, which by default will use same seed_gen as Sample Settings (use existing). You can make a noise layer use a different seed_gen strategy at will, or use a different seed/set of seeds, etc.

The ```mask_optional``` parameter determines where on the initial noise the noise layer should be applied.

| Node | Behavior + Inputs |
|---|---|
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/66487969-669d-47d3-9742-85ae26606903) | [Add]; Adds noise directly on top. <br/> 🟦*noise_weight*: Multiplier for noise layer before being added on top. |
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/52acb25c-9116-4594-b3fb-01b7b15bb79d) | [Add Weighted]; Adds noise, but takes a weighted average between what is already there and itself. <br/> 🟦*noise_weight*: Weight of new noise in the weighted average with existing noise. <br/> 🟦*balance_multipler*: Scale for how much noise_weight should affect existing noise; 1.0 means normal weighted average, and below 1.0 will lessen the weighted reduction by that amount (i.e. if balance_multiplier is set to 0.5 and noise_weight is 0.25, existing noise will only be reduced by 0.125 instead of 0.25, but new noise will be added with the unmodified 0.25 weight). |
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/4feb586e-9920-4f35-8f92-e2e36fabb2df) | [Replace]; Directly replaces existing noise from layers underneath with itself. |


# Samples (download or drag images of the workflows into ComfyUI to instantly load the corresponding workflows!)

NOTE: I've scaled down the gifs to 0.75x size to make them take up less space on the README.

### txt2img

| Result |
|---|
| ![readme_00006](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/b615a4aa-db3e-4b24-b88f-b694e52f6364) |
| Workflow |
| ![t2i_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/6eb47506-b503-482b-9baf-4c238f30a9c2)   |

### txt2img - (prompt travel)

| Result |
|---|
| ![readme_00010](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/c27a2029-2c69-4272-b40f-64408e9e2ea6) |
| Workflow |
| ![t2i_prompttravel_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/e5a72ea1-628d-423e-98ed-f20e1bcc5320) |



### txt2img - 48 frame animation with 16 context_length (Context Options◆Standard Static) + FreeNoise

| Result |
|---|
| ![readme_00012](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/684f6e79-d653-482f-899a-1900dc56cd8f) |
| Workflow |
| ![t2i_context_freenoise_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/9d0e53fa-49d6-483d-a660-3f41d7451002) |


# Old Samples (TODO: update all of these + add new ones when I get sleep)

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



## Known Issues

### Some motion models have visible watermark on resulting images (especially when using mm_sd_v15)

Training data used by the authors of the AnimateDiff paper contained Shutterstock watermarks. Since mm_sd_v15 was finetuned on finer, less drastic movement, the motion module attempts to replicate the transparency of that watermark and does not get blurred away like mm_sd_v14. Using other motion modules, or combinations of them using Advanced KSamplers should alleviate watermark issues.
