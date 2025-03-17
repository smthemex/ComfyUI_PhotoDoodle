# PhotoDoodle
[PhotoDoodle](https://github.com/showlab/PhotoDoodle) it a method about 'Learning Artistic Image Editing from Few-Shot Pairwise Data'，you can use it in comfyUI

# Update
*  if use repo default use nf4 ,fast,and only 12G VRAM ,and not use mmgp,repo模式默认开启nf4量化，得益于新版的diffuser ，可以在不开启mmgp的情况下，轻松使用

# 1.Installation  
-----
  In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_PhotoDoodle
```
# 2.requirements  
----
```
pip install -r requirements.txt
```
* If OOM try  pip install mmgp ，如果OOM ,试一下用mmgp

# 3.checkpoints 
* 3.1 mode use 'flux dev single checkpoints（fp8 or fp16）' or 'repo' or 'unet+ae+comfyui T5XXX' ，三种选择，使用flux dev的fp8或fp16单体模型 或者使用repo，或者使用flux unet+ae+comfy的T5双clip
```
├── ComfyUI/models/diffusion_models/
|     ├── flux1-kj-dev-fp8.safetensors  # if use fp8 unet  11G  unet+vae+clip方法不推荐，因为更容易爆显存
|     ├── flux1-dev-fp8.safetensors  # if use fp8 single 16G  comfy官方单体fp8模型或者flux官方单体模型，正常12G能跑，不用开mmgp
```
* 3.2 more lora download from [here](https://huggingface.co/nicolaus-huang/PhotoDoodle/tree/main)
```
├── ComfyUI/models/loras/
|     ├── pretrain.safetensors  # 必须要
|     ├── skscloudsketch.safetensors  # 选你喜欢的lora
```

# 4 Example
* if use single files，vae must choice "none"  #flux1-dev-fp8.safetensors 16G 单体16G模型，内置clip和vae那种，vae必须选择"none"，不开启mmgp，次要推荐使用
![](https://github.com/smthemex/ComfyUI_PhotoDoodle/blob/main/assets/example.png)
* if use repo 'black-forest-labs/FLUX.1-dev' or C:/youpath/black-forest-labs/FLUX.1-dev  如果使用repo可以用自动下载或本地,不开启mmgp，第一推荐使用
![](https://github.com/smthemex/ComfyUI_PhotoDoodle/blob/main/assets/example_0317.png)
* if use unet #11G 单体unet模型，没有内置clip和vae的，所以必须要连双clip和选vae，可以不开启mmgp，12G可能会OOM
![](https://github.com/smthemex/ComfyUI_PhotoDoodle/blob/main/assets/example_0317A.png)

# 5. Acknowledgments  

1. Thanks to **[Yuxuan Zhang](https://xiaojiu-z.github.io/YuxuanZhang.github.io/)** and **[Hailong Guo](mailto:guohailong@bupt.edu.cn)** for providing the code base.  
2. Thanks to **[Diffusers](https://github.com/huggingface/diffusers)** for the open-source project.

# Citation
```
@misc{huang2025photodoodlelearningartisticimage,
      title={PhotoDoodle: Learning Artistic Image Editing from Few-Shot Pairwise Data}, 
      author={Shijie Huang and Yiren Song and Yuxuan Zhang and Hailong Guo and Xueyin Wang and Mike Zheng Shou and Jiaming Liu},
      year={2025},
      eprint={2502.14397},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.14397}, 
}
```
