# PhotoDoodle
[PhotoDoodle](https://github.com/showlab/PhotoDoodle) it a method about 'Learning Artistic Image Editing from Few-Shot Pairwise Data'，you can use it in comfyUI

# Update
* 改成comfyUI的核心封装方式复现方法/ change to comfyUI core encapsulation method

# 1.Installation  
-----
  In the ./ComfyUI /custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_PhotoDoodle
```
# 2.checkpoints 
```
├── ComfyUI/models/diffusion_models/
|     ├── flux1-kj-dev-fp8.safetensors  # if use fp8 unet  11G  unet+vae+clip方法不推荐，因为更容易爆显存
|     ├── flux1-dev-fp8.safetensors  # if use fp8 single 16G  comfy官方单体fp8模型或者flux官方单体模型，正常12G能跑，不用开mmgp
```
*  more lora download from [here](https://huggingface.co/nicolaus-huang/PhotoDoodle/tree/main)
```
├── ComfyUI/models/loras/
|     ├── pretrain.safetensors  # 必须要
|     ├── skscloudsketch.safetensors  # 选你喜欢的lora
```
# 3 Example

![](https://github.com/smthemex/ComfyUI_PhotoDoodle/blob/main/example_workflows/example.png)


# 4 Citation
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
