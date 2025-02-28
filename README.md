# PhotoDoodle
PhotoDoodle: Learning Artistic Image Editing from Few-Shot Pairwise Data，you can use it in comfyUI

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
If Vram <=24G  pip install mmgp ，这个方法适配的显存是40G，所以4090及以下显卡都要按照mmpg，这样才能跑得快，4090模型加载菜单，可以选0，或者1或者2

# 3.checkpoints 
* 3 mode use flux dev single checkpoints（fp8 or fp16） or repo or unet+ae+comfyui T5XXX ，三种选择，使用flux dev的fp8或fp16单体模型 或者使用repo，或者使用flux unet+ae+comfy的T5双clip
```
├── ComfyUI/models/diffusion_models/
|     ├── flux1-kj-dev-fp8.safetensors  # if use fp8 unet  11G 不推荐，因为更容易爆显存
|     ├── flux1-dev-fp8.safetensors  # if use fp8 single 16G 推荐
```
lora download from [here](https://huggingface.co/nicolaus-huang/PhotoDoodle/tree/main)
```
├── ComfyUI/models/loras/
|     ├── pretrain.safetensors  # 必须要
|     ├── skscloudsketch.safetensors  # 选你喜欢的lora
```

# 4 Example
![](https://github.com/smthemex/ComfyUI_PhotoDoodle/blob/main/assets/example.png)


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
