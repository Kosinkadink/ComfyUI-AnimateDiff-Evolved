# AnimateDiff Evolved Examples

See the Wiki for more detail.

## [Basic txt2vid](./workflows/ade_basic.png)

<details>

<summary>Workflow Image</summary>

![Basic Workflow](./workflows/ade_basic.png)

</details>

Slightly more than the bare minimum.

- Context Options
- Sample Settings
- Apply AnimateDiff Model
- Use Evolved Sampling

![Basic sample](./samples/basic_00001.gif)

## [+ Prompt Travel](./workflows/ade_prompt_travel.png)

<details>

<summary>Workflow Image</summary>

![Prompt Travel Workflow](./workflows/ade_prompt_travel.png)

</details>

Same as above, but with prompt travel.

- Prompt Scheduling [Latents]

![Prompt travel sample](./samples/prompt_travel_00001.gif)

## [+ context_length (32 frames)](./workflows/ade_longer.png)

<details>

<summary>Workflow Image</summary>

![context_length Workflow](./workflows/ade_longer.png)

</details>

A longer generation that uses a sliding window of 16 frames.

![Longer sample](./samples/longer_00001.gif)

## [+ ContextRef](./workflows/ade_contextref.png)

<details>

<summary>Workflow Image</summary>

![ContextRef Workflow](./workflows/ade_contextref.png)

</details>

- Set Context Extras
- Context Extras - ContextRef

### No ContextRef

![ContextRef sample](./samples/contextref_00002.gif)

### ContextRef

![ContextRef sample](./samples/contextref_00001.gif)

## [+ Motion LoRAs](./workflows/ade_motion_loras.png)

<details>

<summary>Workflow Image</summary>

![Motion LoRAs Workflow](./workflows/ade_motion_loras.png)

</details>

- Load AnimateDiff LoRA 🎭🅐🅓
- Zoom Out
- Pan Left

![Motion LoRA sample](./samples/motionlora_00001.gif)

## [+ Latent Upscale](./workflows/ade_upscale_1.png)

<details>

<summary>Workflow Image</summary>

![Latent Upscale Workflow](./workflows/ade_upscale_1.png)

</details>

- Classic Upscaling
  
![Upscale sample before](./samples/upscale_base_00001.gif) ![Upscale sample after](./samples/upscale_up_00001.gif)