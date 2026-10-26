# -*- coding: UTF-8 -*-
"""
PhotoDoodle native-DiT refactor.

Goal state: native loaders -> this patch node -> native KSampler / VAEDecode.
Plugin dependencies after this refactor: torch only (no diffusers / transformers).

How PhotoDoodle works (official pipeline_pe_clone.py):
  - reference image is VAE-encoded to *scaled* flux latents, packed to tokens and
    CONCATENATED to the noisy latent tokens along the sequence dim
    (order: [noise tokens, reference tokens]).
  - position encoding is CLONED: reference tokens reuse the exact (0, y, x) ids of
    the target tokens (that is the "PE-clone" trick).
  - the reference half never integrates: after every scheduler step it is reset to
    the clean latents (mask trick). Note the shift/mu for the sigma schedule is
    computed with the DOUBLED sequence length (target + reference tokens).

ComfyUI native equivalents (no comfy source modification):
  - comfy.ldm.flux.model.Flux already supports reference latents: ref tokens are
    appended after the target tokens and their ids are built by process_img().
    Method "index" with params.ref_index_scale = 0.0 yields ids (0, y, x) for a
    same-size reference == exact PE-clone.
  - reference tokens are cropped from the model output (out[:, :img_tokens]) and
    the ref latent is static conditioning -> equivalent to the official
    "reset every step" mask, so no custom sampler is needed.
  - the doubled-seq-len mu is applied by setting model_sampling shift = mu(2N).
"""

import comfy.model_sampling
import nodes
import node_helpers

MAX_RESOLUTION = nodes.MAX_RESOLUTION


def calculate_mu(image_seq_len, base_seq_len=256, max_seq_len=4096,
                 base_shift=0.5, max_shift=1.15):
    # same formula as official pipeline_pe_clone.calculate_shift
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class PhotoDoodleNativePatch:
    """Attaches the reference latent (PE-clone) to the conditioning and patches
    the model: ref_index_scale=0 (position clone) + doubled-seq-len sigma shift."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
                "base_seq_len": ("INT", {"default": 256, "min": 16, "max": 8192, "step": 16, "advanced": True}),
                "max_seq_len": ("INT", {"default": 4096, "min": 16, "max": 16384, "step": 16, "advanced": True}),
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "advanced": True}),
                "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 10.0, "step": 0.01, "advanced": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "conditioning")
    FUNCTION = "patch"
    CATEGORY = "PhotoDoodle"
    DESCRIPTION = ("PhotoDoodle native patch. The image is resized to width x height, "
                   "VAE-encoded and attached as a PE-cloned reference latent. "
                   "Sigma shift is computed with the doubled (target+ref) token count, "
                   "matching the official pipeline. Use scheduler=normal, cfg=1.0.")

    def patch(self, model, conditioning, latent,
              base_seq_len, max_seq_len, base_shift, max_shift):

        conditioning = node_helpers.conditioning_set_values(
            conditioning,
            {"reference_latents": [latent["samples"]], "reference_latents_method": "index"},
            append=True,
        )

        #  model patches (work on a clone, like comfy's own ModelSamplingFlux node)
        m = model.clone()
        diffusion_model = m.model.diffusion_model
        if hasattr(diffusion_model, "params"):
            # PhotoDoodle trains with cloned positions: t-axis of ref tokens = 0
            diffusion_model.params.ref_index_scale = 0.0
            diffusion_model.params.default_ref_method = "index"
        height,width=latent["samples"].shape[2:4]
        print(f"PhotoDoodleNativePatch:  height={height}, width={width}")
        # official uses the DOUBLED sequence length for the dynamic shift
        img_seq_len = (height // 2) * (width // 2)
        mu = calculate_mu(2 * img_seq_len, base_seq_len, max_seq_len, base_shift, max_shift)

        class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingFlux,
                                    comfy.model_sampling.CONST):
            pass

        model_sampling = ModelSamplingAdvanced(m.model.model_config)
        model_sampling.set_parameters(shift=mu)
        m.add_object_patch("model_sampling", model_sampling)

        return (m,conditioning)

NODE_CLASS_MAPPINGS = {
    "PhotoDoodleNativePatch": PhotoDoodleNativePatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoDoodleNativePatch": "PhotoDoodle Native Patch (PE-Clone)",
}
