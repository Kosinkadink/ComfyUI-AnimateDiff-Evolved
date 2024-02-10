import torch

from .utils_model import BetaSchedules, SigmaSchedule, ModelSamplingType, ModelSamplingConfig, InterpolationMethod


def validate_sigma_schedule_compatibility(schedule_A: SigmaSchedule, schedule_B: SigmaSchedule,
                                          name_a: str="sigma_schedule_A", name_b: str="sigma_schedule_B"):
    if schedule_A.total_sigmas() != schedule_B.total_sigmas():
            raise Exception(f"Weighted Average cannot be taken of Sigma Schedules that do not have the same amount of sigmas; " +
                            f"{name_a} has {schedule_A.total_sigmas()} sigmas (lcm={schedule_A.is_lcm()}), " +
                            f"{name_b} has {schedule_B.total_sigmas()} sigmas (lcm={schedule_B.is_lcm()}).")


class SigmaScheduleNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "beta_schedule": (BetaSchedules.ALIAS_ACTIVE_LIST,),
            }
        }
    
    RETURN_TYPES = ("SIGMA_SCHEDULE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings/sigma schedule"
    FUNCTION = "get_sigma_schedule"

    def get_sigma_schedule(self, beta_schedule: str):
        model_type = ModelSamplingType.from_alias(ModelSamplingType.EPS)
        new_model_sampling = BetaSchedules._to_model_sampling(alias=beta_schedule,
                                                              model_type=model_type)
        return (SigmaSchedule(model_sampling=new_model_sampling, model_type=model_type),)


class RawSigmaScheduleNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_beta_schedule": (BetaSchedules.RAW_BETA_SCHEDULE_LIST,),
                "linear_start": ("FLOAT", {"default": 0.00085, "min": 0.0, "max": 1.0, "step": 0.000001}),
                "linear_end": ("FLOAT", {"default": 0.012, "min": 0.0, "max": 1.0, "step": 0.000001}),
                #"cosine_s": ("FLOAT", {"default": 8e-3, "min": 0.0, "max": 1.0, "step": 0.000001}),
                "sampling": (ModelSamplingType._FULL_LIST,),
                "lcm_original_timesteps": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "lcm_zsnr": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("SIGMA_SCHEDULE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings/sigma schedule"
    FUNCTION = "get_sigma_schedule"

    def get_sigma_schedule(self, raw_beta_schedule: str, linear_start: float, linear_end: float,# cosine_s: float,
                           sampling: str, lcm_original_timesteps: int, lcm_zsnr: bool):
        new_config = ModelSamplingConfig(beta_schedule=raw_beta_schedule, linear_start=linear_start, linear_end=linear_end)
        if sampling != ModelSamplingType.LCM:
            lcm_original_timesteps=None
            lcm_zsnr=False
        model_type = ModelSamplingType.from_alias(sampling)    
        new_model_sampling = BetaSchedules._to_model_sampling(alias=BetaSchedules.AUTOSELECT, model_type=model_type, config_override=new_config, original_timesteps=lcm_original_timesteps)
        if lcm_zsnr:
            SigmaSchedule.apply_zsnr(new_model_sampling=new_model_sampling)
        return (SigmaSchedule(model_sampling=new_model_sampling, model_type=model_type),)


class WeightedAverageSigmaScheduleNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "schedule_A": ("SIGMA_SCHEDULE",),
                "schedule_B": ("SIGMA_SCHEDULE",),
                "weight_A": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }
    
    RETURN_TYPES = ("SIGMA_SCHEDULE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings/sigma schedule"
    FUNCTION = "get_sigma_schedule"

    def get_sigma_schedule(self, schedule_A: SigmaSchedule, schedule_B: SigmaSchedule, weight_A: float):
        validate_sigma_schedule_compatibility(schedule_A, schedule_B)
        new_sigmas = schedule_A.model_sampling.sigmas * weight_A + schedule_B.model_sampling.sigmas * (1-weight_A)
        combo_schedule = schedule_A.clone()
        combo_schedule.model_sampling.set_sigmas(new_sigmas)
        return (combo_schedule,)


class InterpolatedWeightedAverageSigmaScheduleNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "schedule_A": ("SIGMA_SCHEDULE",),
                "schedule_B": ("SIGMA_SCHEDULE",),
                "weight_A_Start": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "weight_A_End": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "interpolation": (InterpolationMethod._LIST,),
            }
        }
    
    RETURN_TYPES = ("SIGMA_SCHEDULE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings/sigma schedule"
    FUNCTION = "get_sigma_schedule"

    def get_sigma_schedule(self, schedule_A: SigmaSchedule, schedule_B: SigmaSchedule,
                           weight_A_Start: float, weight_A_End: float, interpolation: str):
        validate_sigma_schedule_compatibility(schedule_A, schedule_B)
        # get reverse weights, since sigmas are currently reversed
        weights = InterpolationMethod.get_weights(num_from=weight_A_Start, num_to=weight_A_End,
                                                  length=schedule_A.total_sigmas(), method=interpolation, reverse=True)
        weights = weights.to(schedule_A.model_sampling.sigmas.dtype).to(schedule_A.model_sampling.sigmas.device)
        new_sigmas = schedule_A.model_sampling.sigmas * weights + schedule_B.model_sampling.sigmas * (1.0-weights)
        combo_schedule = schedule_A.clone()
        combo_schedule.model_sampling.set_sigmas(new_sigmas)
        return (combo_schedule,)


class SplitAndCombineSigmaScheduleNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "schedule_Start": ("SIGMA_SCHEDULE",),
                "schedule_End": ("SIGMA_SCHEDULE",),
                "idx_split_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001})
            }
        }
    
    RETURN_TYPES = ("SIGMA_SCHEDULE",)
    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings/sigma schedule"
    FUNCTION = "get_sigma_schedule"

    def get_sigma_schedule(self, schedule_Start: SigmaSchedule, schedule_End: SigmaSchedule, idx_split_percent: float):
        validate_sigma_schedule_compatibility(schedule_Start, schedule_End)
        # first, calculate index to act as split; get diff from 1.0 since sigmas are flipped at this stage
        idx = int((1.0-idx_split_percent) * schedule_Start.total_sigmas())
        new_sigmas = torch.cat([schedule_End.model_sampling.sigmas[:idx], schedule_Start.model_sampling.sigmas[idx:]], dim=0)
        new_schedule = schedule_Start.clone()
        new_schedule.model_sampling.set_sigmas(new_sigmas)
        return (new_schedule,)
