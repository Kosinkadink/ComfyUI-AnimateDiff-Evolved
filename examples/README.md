# AnimateDiff Evolved Examples

## Basic txt2vid

Slightly more than the bare minimum.

- Context Options
- Sample Settings
- Apply AnimateDiff Model
- Use Evolved Sampling

|                          Workflow                          |                   Sample                   |
| :--------------------------------------------------------: | :----------------------------------------: |
| ![Basic txt2vid workflow image](./workflows/ade_basic.png) | ![Basic sample](./samples/basic_00001.gif) |

## + Prompt Travel

Same as above, but with prompt travel.

- Prompt Scheduling [Latents]

|                              Workflow                              |                           Sample                           |
| :----------------------------------------------------------------: | :--------------------------------------------------------: |
| ![Prompt Travel workflow image](./workflows/ade_prompt_travel.png) | ![Prompt travel sample](./samples/prompt_travel_00001.gif) |

## + context_length (32 frames)

A longer generation that uses a sliding window of 16 frames.

|                        Workflow                        |                    Sample                    |
| :----------------------------------------------------: | :------------------------------------------: |
| ![32 Frame workflow image](./workflows/ade_longer.png) | ![Longer sample](./samples/longer_00001.gif) |

## + ContextRef

- Set Context Extras
- Context Extras - ContextRef

|                           Workflow                           |                        Sample                        |
| :----------------------------------------------------------: | :--------------------------------------------------: |
| ![ContextRef workflow image](./workflows/ade_contextref.png) | ![ContextRef sample](./samples/contextref_00001.gif) |

## + Motion LoRAs

- Load AnimateDiff LoRA 🎭🅐🅓
- Zoom Out
- Pan Left

|                            Workflow                            |                        Sample                         |
| :------------------------------------------------------------: | :---------------------------------------------------: |
| ![Motion LoRA workflow image](./workflows/ade_motionloras.png) | ![Motion LoRA sample](./samples/motionlora_00001.gif) |

## + Latent Upscale

- Classic Upscaling
  
|                        Workflow                        |                                                       Sample                                                       |
| :----------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------: |
| ![Upscale workflow image](./workflows/ade_upscale_1.png) | ![Upscale sample before](./samples/upscale_base_00001.gif) ![Upscale sample after](./samples/upscale_up_00001.gif) |
