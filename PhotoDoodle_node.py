# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import numpy as np
from safetensors.torch import load_file
from diffusers import AutoencoderKL,FluxTransformer2DModel
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig,T5EncoderModel
from .node_utils import tensor2pil_list, load_images,cleanup
from .src.pipeline_pe_clone import FluxPipeline
from .src.pipeline_pe_clone_orgin import FluxPipeline as FluxPipeline_orgin

import folder_paths
from comfy.utils import ProgressBar
MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")


class PhotoDoodle_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        
        return {
            "required": {
                "flux_unet": (["none"] + folder_paths.get_filename_list("diffusion_models"),),
                "vae": (["none"] + folder_paths.get_filename_list("vae"),),
                "pre_lora": (["none"] + [i for i in folder_paths.get_filename_list("loras") if "pre" in i],),
                "loras": (["none"] + folder_paths.get_filename_list("loras"),),
                "flux_repo":("STRING", {"default": "", "multiline": False}),
                "quantization":(["none","fp8","nf4",],),
                "quantize_T5":("BOOLEAN",{"default":False}),
                "use_mmgp":("BOOLEAN",{"default":False}),
                "profile_number":([1,2,3,4,5],)
            },
        }

    RETURN_TYPES = ("MODEL_PhotoDoodle",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "PhotoDoodle"

    
    def loader_main(self,flux_unet,vae,pre_lora,loras,flux_repo,quantization,quantize_T5,use_mmgp,profile_number):
       
        flux_repo_local=os.path.join(current_node_path, 'src/FLUX.1-dev')   
        print("***********Load model ***********")
        #load model
        
        if pre_lora != "none":
            pre_lora_path = folder_paths.get_full_path("loras", pre_lora)
        else:
            raise ValueError("No model selected")
    
        if loras != "none":
            lora_path = folder_paths.get_full_path("loras", loras)
        else:
            raise ValueError("No model selected")
        need_clip=False
        if  not flux_repo:
            if flux_unet != "none":
                flux_transformer_path = folder_paths.get_full_path("diffusion_models", flux_unet)
            else:
                raise ValueError("No model selected")
            if vae != "none":
                need_clip=True
                vae_path = folder_paths.get_full_path("vae", vae)
                vae_config=os.path.join(flux_repo_local, 'vae')
                ae = AutoencoderKL.from_single_file(vae_path,config=vae_config, torch_dtype=torch.bfloat16)
                config_file = os.path.join(flux_repo_local,"transformer/config.json")

                if quantization=="fp8":
                    if os.path.splitext(flux_transformer_path)[-1] == ".pt":
                        transformer = torch.load(flux_transformer_path)
                        transformer.eval()
                    else:
                        transformer = FluxTransformer2DModel.from_single_file(flux_transformer_path, config=config_file,
                                                                            torch_dtype=torch.bfloat16)
                else:
                    t_state_dict=load_file(flux_transformer_path)
                    unet_config = FluxTransformer2DModel.load_config(config_file)
                    transformer = FluxTransformer2DModel.from_config(unet_config).to(torch.bfloat16)
                    transformer.load_state_dict(t_state_dict, strict=False)
                    del t_state_dict
                    cleanup()
                pipeline = FluxPipeline.from_pretrained(
                    flux_repo_local,
                    vae=ae,
                    transformer=transformer,
                    torch_dtype=torch.bfloat16,
                    )     
            else:
                pipeline = FluxPipeline_orgin.from_single_file(
                            flux_transformer_path,
                            config=flux_repo_local,
                            torch_dtype=torch.bfloat16,
                            )
            
        else:
            # flux_repo='F:/test/ComfyUI/models/diffusers/black-forest-labs/FLUX.1-dev'
            if quantization=="none":
                if quantize_T5:
                    quant_config = TransformersBitsAndBytesConfig(load_in_8bit=True,) # 8bit default
                    text_encoder_2_8bit = T5EncoderModel.from_pretrained(
                    flux_repo,
                    subfolder="text_encoder_2",
                    quantization_config=quant_config,
                    torch_dtype=torch.float16,
                )
                    pipeline =FluxPipeline_orgin.from_pretrained(flux_repo,text_encoder_2=text_encoder_2_8bit,torch_dtype=torch.bfloat16,)
                else:
                    pipeline =FluxPipeline_orgin.from_pretrained(flux_repo,torch_dtype=torch.bfloat16,)
                    
            else:
                if quantize_T5:
                        quant_config = TransformersBitsAndBytesConfig(load_in_8bit=True,) # 8bit 
                        text_encoder_2_8bit = T5EncoderModel.from_pretrained(
                        flux_repo,
                        subfolder="text_encoder_2",
                        quantization_config=quant_config,
                        torch_dtype=torch.float16,
                    )
                if quantization=="fp8":
                    transformer = FluxTransformer2DModel.from_pretrained(
                        flux_repo,
                        subfolder="transformer",
                        quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True,),
                        torch_dtype=torch.bfloat16,
                    )
                   
                else: #nf4
                    transformer = FluxTransformer2DModel.from_pretrained(
                        flux_repo,
                        subfolder="transformer",
                        quantization_config=DiffusersBitsAndBytesConfig(
                            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
                        ),
                        torch_dtype=torch.bfloat16,)
                if quantize_T5:
                    pipeline =FluxPipeline_orgin.from_pretrained(flux_repo,transformer=transformer,text_encoder_2=text_encoder_2_8bit,torch_dtype=torch.bfloat16,)
                else: #nf4
                    pipeline =FluxPipeline_orgin.from_pretrained(flux_repo,transformer=transformer,torch_dtype=torch.bfloat16,)

        # Load and fuse base LoRA weights
        print("***********  Load and fuse base LoRA weights ***********")
        pipeline.load_lora_weights(lora_path)
        pipeline.fuse_lora()
        pipeline.unload_lora_weights()
        pipeline.load_lora_weights(pre_lora_path)
        pipeline.enable_model_cpu_offload()

        print("***********Load model done ***********")

        cleanup()

        if use_mmgp:
            from mmgp import offload as offloadobj
            pipe = {"transformer": pipeline, }
            # offloadobj.profile(pipe, quantizeTransformer = False,  profile_no = 1 ) # uncomment this line and comment the previous one if you have 24 GB of VRAM and wants faster generation  
            offloadobj.profile(pipe, quantizeTransformer = False,  extraModelsToQuantize = [], profile_no = profile_number, ) 
            del pipe
        return ({"pipeline":pipeline,"need_clip":need_clip},)



class PhotoDoodle_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_PhotoDoodle",),
                "images": ("IMAGE",),  
                "prompt": ("STRING", {"default": "add a halo and wings for the cat by sksmagiceffects", "multiline": True}),
                 "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED, "step": 1, "display": "number"}),
                "width": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 64, "display": "number"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1024, "step": 1, "display": "number"}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "max_sequence_length": ("INT", {"default": 512, "min": 128, "max": 512, "step": 1, "display": "number"}),
                },
            "optional": { "clip":("CLIP",),},
            }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image",)
    FUNCTION = "sampler_main"
    CATEGORY = "PhotoDoodle"

    def sampler_main(self, model,images,prompt,seed, width, height,steps,guidance_scale, max_sequence_length,**kwargs):

        need_clip=model.get("need_clip")
        pipeline=model.get("pipeline")
        clip=kwargs.get("clip")
        # load model
        #model.to(device)
        if need_clip:
            if clip is None:
                raise ValueError("No clip selected")
            # fix pooled of comfyclip
            tokens = clip.tokenize(prompt)
            tokens["t5xxl"] = clip.tokenize(prompt)["t5xxl"]
            prompt_embeds = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True).pop("cond")
            tokens["l"] = clip.tokenize(prompt)["l"]
            pooled_prompt_embeds = clip.encode_from_tokens(tokens,  return_dict=True).pop("pooled_output")
            #prompt_embeds,pooled_prompt_embeds=clip.encode_from_tokens( clip.tokenize(prompt,return_word_ids=True),return_pooled=True)
            #print(prompt_embeds.shape,pooled_prompt_embeds.shape)
            prompt_embeds=prompt_embeds.to(device,torch.bfloat16)
            pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.bfloat16)
            prompt=None
            del clip

        else:
            prompt_embeds,pooled_prompt_embeds=None,None

        cleanup()
        condition_image_list = tensor2pil_list(images, width, height)
        total_images=len(condition_image_list)
        pbar = ProgressBar(total_images)
        img_list=[]

        for i, condition_image in enumerate(condition_image_list):
            result = pipeline(
                prompt=prompt,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                condition_image=condition_image,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                max_sequence_length=max_sequence_length,
                generator = torch.Generator(device=device).manual_seed(seed)
            ).images[0]
            pbar.update_absolute(i, total_images)
            img_list.append(result)
            
        cleanup()
        return (load_images(img_list),)


NODE_CLASS_MAPPINGS = {
    "PhotoDoodle_Loader": PhotoDoodle_Loader,
    "PhotoDoodle_Sampler": PhotoDoodle_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoDoodle_Loader": "PhotoDoodle_Loader",
    "PhotoDoodle_Sampler": "PhotoDoodle_Sampler",
}
