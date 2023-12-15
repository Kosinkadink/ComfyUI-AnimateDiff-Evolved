from .sample_settings import SeedNoiseType

class SampleSettingsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed_noise": (SeedNoiseType.LIST),
            },
            "optional": {
                "noise_layers": ("NOISE_LAYERS", ),
            }
        }
    
    RETURN_TYPES = ("SAMPLE_SETTINGS")

