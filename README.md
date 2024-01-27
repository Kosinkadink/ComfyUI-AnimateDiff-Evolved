# AnimateDiff for ComfyUI

Improved [AnimateDiff](https://github.com/guoyww/AnimateDiff/) integration for ComfyUI, as well as advanced sampling options dubbed Evolved Sampling usable outside of AnimateDiff. Please read the AnimateDiff repo README and Wiki for more information about how it works at its core.

AnimateDiff workflows will often make use of these helpful node packs:
- [ComfyUI_FizzNodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes) for prompt-travel functionality with the BatchPromptSchedule node. Maintained by FizzleDorf.
- [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet) for making ControlNets work with Context Options and controlling which latents should be affected by the ControlNet inputs. Includes SparseCtrl support. Maintained by me.
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) for loading videos, combining images into videos, and doing various image/latent operations like appending, splitting, duplicating, selecting, or counting. Actively maintained by AustinMroz and I.
- [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) for ControlNet preprocessors not present in vanilla ComfyUI. Maintained by Fannovel16.
- [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) for IPAdapter support. Maintained by cubiq (matt3o).

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
- Mixable Motion LoRAs from [original AnimateDiff repository](https://github.com/guoyww/animatediff/) implemented. Caveat: only really work on v2-based motion models like ```mm_sd_v15_v2```, ```mm-p_0.5.pth```, and ```mm-p_0.75.pth```
- Prompt travel using BatchPromptSchedule node from [ComfyUI_FizzNodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes)
- Scale and Effect multival inputs to control motion amount and motion model influence on generation.
     - Can be float, list of floats, or masks
- Custom noise scheduling via Noise Types, Noise Layers, and seed_override/seed_offset/batch_offset in Sample Settings and related nodes
- AnimateDiff model v1/v2/v3 support
- Using multiple motion models at once via Gen2 nodes (each supporting 
- [HotshotXL](https://huggingface.co/hotshotco/Hotshot-XL/tree/main) support (an SDXL motion module arch), ```hsxl_temporal_layers.safetensors```.
     - NOTE: You will need to use ```autoselect``` or ```linear (HotshotXL/default)``` beta_schedule, the sweetspot for context_length or total frames (when not using context) is 8 frames, and you will need to use an SDXL checkpoint.
- AnimateDiff-SDXL support, with corresponding model. Currently, a beta version is out, which you can find info about at [AnimateDiff](https://github.com/guoyww/AnimateDiff/).
     - NOTE: You will need to use ```autoselect``` or ```linear (AnimateDiff-SDXL)``` beta_schedule. Other than that, same rules of thumb apply to AnimateDiff-SDXL as AnimateDiff.
- AnimateDiff Keyframes to change Scale and Effect at different points in the sampling process.
- fp8 support; requires newest ComfyUI and torch >= 2.1 (decreases VRAM usage, but changes outputs)
- Mac M1/M2/M3 support
- Usage of Context Options and Sample Settings outside of AnimateDiff via Gen2 Use Evolved Sampling node

## Upcoming Features
- Maskable SD LoRA (and perhaps maskable SD Models as well)
- [PIA](https://github.com/open-mmlab/PIA) support
- Motion LoRA training (experimental)
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
- 🟨*ad_keyframes*: Allows scheduling of ```scale_multival``` and ```scale_effect``` inputs across sampling timesteps.
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
| ![anim__00012](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/7aee4ccb-b669-42fd-a1b5-2005003d5f8d) <br/> (latent count: 64, view_length: 16, view_overlap: 4, Context Options◆Standard Static, context_length: 16, context_overlap: 8, total steps: 20) |

| Node | Inputs |
|---|---|
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/4b22c73f-99cb-4781-bd33-e1b3db848207) | 🟦*view_length*: Amount of latents in context to pass into motion model at a time.<br/> 🟦*view_overlap*: Minimum common latents between adjacent windows.<br/> 🟦*fuse_method*: Method for averaging results of windows.<br/> |

### View Options◆Standard Uniform
| Behavior |
|---|
| ![anim__00015](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/faa2cd26-9f94-4fce-90b2-8acec84b444e ) <br/> (latent count: 64, view_length: 16, view_overlap: 4, view_stride: 1, Context Options◆Standard Static, context_length: 16, context_overlap: 8, total steps: 20) |

| Node | Inputs |
|---|---|
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/bbf017e6-3545-4043-ba41-fcbe2f54496a) | 🟦*view_length*: Amount of latents in context to pass into motion model at a time.<br/> 🟦*view_overlap*: Minimum common latents between adjacent windows.<br/> 🟦*view_stride*: Maximum 2^(stride-1) distance between adjacent latents.<br/> 🟦*fuse_method*: Method for averaging results of windows.<br/> |

### View Options◆Looped Uniform
| Behavior |
|---|
| ![anim__00016](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/8922b44b-cb19-4b2a-8486-2df8a46bf573) <br/> (latent count: 64, view_length: 16, view_overlap: 4, view_stride: 1, closed_loop: False, Context Options◆Standard Static, context_length: 16, context_overlap: 8, total steps: 20) |
| NOTE: this one is probably not going to come out looking well unless you are using this for a very specific reason. |

| Node | Inputs |
|---|---|
| ![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/c58fe4d4-81a8-436b-8028-9e81c2ace18a) | 🟦*view_length*: Amount of latents in context to pass into motion model at a time.<br/> 🟦*view_overlap*: Minimum common latents between adjacent windows.<br/> 🟦*view_stride*: Maximum 2^(stride-1) distance between adjacent latents.<br/> 🟦*closed_loop*: When True, adds additional windows to enhance looping.<br/> 🟦*use_on_equal_length*: When True, allows context to be used when latent count matches context_length - allows loops to be made when latent count == context_length.<br/> 🟦*fuse_method*: Method for averaging results of windows.<br/> |




# Core Nodes



# Notable Updates
### (December 6th, 2023) Massive rewrite of code 

I just released a massive rework of the code that I've been working on the past week. Changes are almost all under the hood, and everything should still look the same generation-wise and performance-wise. ComfyUI design patterns and model management is used where possible now. If you experience any issues you did not have before, please report them so I can fix them quickly!
Notable changes:
- Slightly lower VRAM usage (0.3-0.8GB) depending on workflow
- Motion model caching - speeds up consecutive sampling
- fp8 support (by casting in places that need to be casted)
- Model patches (like LCM) can be applied properly (no guarantees on improvements in generations though, might take some investigation to figure out why v2 models look weird with LCM)
- dtype and device mismatch edge cases should now be fixed
- Additional 'use existing' beta schedule to allow any ModelSampling nodes to take effect - will use beta schedule as the ModelSampling patch overwise

# Features:
- Compatible with a variety of samplers, vanilla KSampler nodes and KSampler (Effiecient) nodes.
- ControlNet support - both per-frame, or "interpolating" between frames; can kind of use this as img2video (see workflows below)
- Infinite animation length support using sliding context windows **(introduced 9/17/23)**
- Mixable Motion LoRAs from [original AnimateDiff repository](https://github.com/guoyww/animatediff/) implemented. Caveat: only really work on v2-based motion models like ```mm_sd_v15_v2```, ```mm-p_0.5.pth```, and ```mm-p_0.75.pth``` **(introduced 9/25/23)**
- Prompt travel using BatchPromptSchedule node from [ComfyUI_FizzNodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes) **(working since 9/27/23)**
- [HotshotXL](https://huggingface.co/hotshotco/Hotshot-XL/tree/main) support (an SDXL motion module arch), ```hsxl_temporal_layers.safetensors``` **(working since 10/05/23)** NOTE: You will need to use ```linear (HotshotXL/default)``` beta_schedule, the sweetspot for context_length or total frames (when not using context) is 8 frames, and you will need to use an SDXL checkpoint. Will add more documentation and example workflows soon when I have some time between working on features/other nodes.
- Motion scaling and other motion model settings to influence motion amount **(introduced 10/30/23)**
- Motion scaling masks in Motion Model Settings, allowing to choose how much motion to apply per frame or per area of each frame **(introduced 11/08/23)**. Can be used alongside inpainting (gradient masks supported for AnimateDiff masking)
- AnimateDiff-SDXL support, with corresponding model. **(introduced 11/10/23)**. Currently, a beta version is out, which you can find info about at [AnimateDiff](https://github.com/guoyww/AnimateDiff/). NOTE: You will need to use ```linear (AnimateDiff-SDXL)``` beta_schedule. Other than that, same rules of thumb apply to AnimateDiff-SDXL as AnimateDiff.
- fp8 support: requires newest ComfyUI and torch >= 2.1 **(introduced 12/06/23)**.
- AnimateDiff v3 motion model support **(introduced 12/15/23)**.

# Upcoming features:
- Alternate context schedulers and context types (in progress)

# Core Nodes:

## AnimateDiff Loader
![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/7dba5a1c-8c29-4d8c-9778-17e217af81b9)



The ***only required node to use AnimateDiff***, the Loader outputs a model that will perform AnimateDiff functionality when passed into a sampling node.

Inputs:
- model: model to setup for AnimateDiff usage. ***Must be a SD1.5-derived model.***
- context_options: optional context window to use while sampling; if passed in, total animation length has no limit. If not passed in, animation length will be limited to either 24 or 32 frames, depending on motion model.
- motion_lora: optional motion LoRA input; if passed in, can influence movement.
- motion_model_settings: optional settings to influence motion model.
- model_name: motion model to use with AnimateDiff.
- beta_schedule: noise scheduler for SD. ```sqrt_linear``` is the intended way to use AnimateDiff, with expected saturation. However, ```linear``` can give useful results as well, so feel free to experiment.
- motion_scale: change motion amount generated by motion model - if less than 1, less motion; if greater than 1, more motion.

Outputs:
- MODEL: model injected to perform AnimateDiff functions

### Usage
To use, just plug in your model into the AnimateDiff Loader. When the output model (and any derivative of it in this pathway) is passed into a sampling node, AnimateDiff will do its thing.

The desired animation length is determined by the latents passed into the sampler. **With context_options connected, there is no limit to the amount of latents you can pass in, AKA unlimited animation length.** When no context_options are connected, the sweetspot is 16 latents passed in for best results, with a limit of 24 or 32 based on motion model loaded. **These same rules apply to Uniform Context Option's context_length**.

You can also connect AnimateDiff LoRA Loader nodes to influence the overall movement in the image - currently, only works well on motion v2-based models.

[Simplest Usage]
![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/95c5edc1-3d48-4f03-9f04-4d4ce54c8602)
[All Possible Connections Usage]
![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/22ffaf8a-bda9-428d-be11-163cb75feae4)



## Uniform Context Options
TODO: fill this out
![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/0327521c-15df-46d8-bee4-6c80b2d7d02d)



## AnimateDiff LoRA Loader
![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/11159b61-7077-4cb1-864c-078bfe82ece3)

Allows plugging in Motion LoRAs into motion models. Current Motion LoRAs only properly support v2-based motion models. Does not affect sampling speed, as the values are frozen after model load. **If you experience slowdowns for using LoRAs, please open an issue so I can resolve it.** Currently, the three models that I know are v2-based are ```mm_sd_v15_v2```, ```mm-p_0.5.pth```, and ```mm-p_0.75.pth```.

Inputs:
- lora_name: name of Motion LoRAs placed in ```ComfyUI/custom_node/ComfyUI-AnimateDiff-Evolved/motion-lora``` directory.
- strength: how strong (or weak) effect of Motion LoRA should be. Too high a value can lead to artifacts in final render.
- prev_motion_lora: optional input allowing to stack LoRAs together.

Outputs:
- MOTION_LORA: motion_lora object storing the names of all the LoRAs that were chained behind it - can be plugged into the back of another AnimateDiff LoRA Loader, or into AniamateDiff Loader's motion_lora input.

[Simplest Usage]
![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/520fb048-1d36-4ec0-a072-532817eafdc0)
[Chaining Multiple Motion LoRAs]
![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/9c68fcb1-4573-4ce7-a9a5-4d2b1163d762)



## Motion Model Settings
![image](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/12887803-8a4f-443b-9c73-82d4b8fc7c75)

Additional tweaks to the internals of the motion models. The Advanced settings will take a whole guide to explain, and I currently do not have the time for that. Instead, I'll focus on the simple settings.

Inputs:
- motion_pe_stretch: used to decrease the amount of motion by stretching (and interpolating) between the positional encoders (PEs). TL;DR: number go up, animation slow down. Number up too much, animation begins to vibrate (vibration artifacts).

Outputs:
- MOTION_MODEL_SETTINGS: motion_model_settings object to be plugged into an AnimateDiff Loader.


## Samples (download or drag images of the workflows into ComfyUI to instantly load the corresponding workflows!)

### txt2img

![t2i_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/b1374343-7b86-453f-b6f5-9717fd8b09aa)

![aaa_readme_00001_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/adf2d591-85c4-4d84-9a6f-f7296b5b7f76)

[aaa_readme_00003_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/974c77ea-47a2-422f-bea5-b080549fb17c)



### txt2img - (prompt travel)

![t2i_prompttravel_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/54424a3b-fb05-4119-811a-727ebcf4969a)

![aaa_readme_00008_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/8911cd93-be2a-4e20-a90b-b356fb2dbc59)

[aaa_readme_00010_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/294c41fb-bd1f-4641-befe-b4fc0dc480c3)



### txt2img - 48 frame animation with 16 context_length (uniform)

![t2i_context_wf](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/a9232105-360d-4947-b88c-78f933af4d5a)

![aaa_readme_00004_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/1c5433c2-e368-48ff-a3a7-dfee7b9cc7a8)

[aaa_readme_00006_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/f724367a-68e6-4e83-a23f-20abf692ce0c)



### txt2img - (prompt travel) 48 frame animation with 16 context_length (uniform)

![t2i_context_promptravel](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/d2b9cfd1-3c2f-4660-86ce-0a60db1ad4ad)

![aaa_readme_00001_](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/5aad2768-1b16-4e2d-a26a-89f3c1a8954f)

[aaa_readme_00002_.webm](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/assets/7365912/129a95da-d541-489a-8eb4-d734fe22e90c)


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
