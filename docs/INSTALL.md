# 安装与固定版本

每个模型使用独立Python环境。`verification/software_versions.json`记录已成功服务器的实际版本与官方源码commit；先clone官方源码并checkout该commit，再按仓库安装说明部署。原始install脚本在reference中，不能直接运行其中他人路径。

官方源码入口：

- Hunyuan：https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5
- Wan加速框架：https://github.com/ModelTC/LightX2V
- LTX：https://github.com/Lightricks/LTX-2
- SANA：https://github.com/NVlabs/Sana

## 权重目录

Hunyuan、Wan分别使用官方原生完整权重目录，不使用Diffusers转换权重。

LTX weights目录应包含：

```text
diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors
text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
vae/ltx-2.5-video-vae-bf16.safetensors
vae/ltx-2.5-audio-vae-bf16.safetensors
latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
```

SANA weights目录应包含：

```text
transformer/checkpoints/SANA_Video_2.0_5B_720p.pth
LTX-2.3-Diffusers/vae/config.json
LTX-2.3-Diffusers/vae/diffusion_pytorch_model.safetensors
```

SANA额外参数`--text-encoder`指向包含config.json、tokenizer文件和两片model safetensors的本地Gemma目录。已使用的固定快照为 `Efficient-Large-Model/gemma-2-2b-it@569d9809d0c8b6722d4d31b5a77a2ec7a400650a`。SANA transformer revision `f2d95fa06400f186fd1b077d12c69c8ac58aba48`；VAE revision `362acdf779d42e785fb26910c32254e9458e78c2`。自行通过官方渠道获取有权访问的组件。

## SANA编译头文件

Triton需要与运行Python版本一致的完整开发头文件，不能拿Python3.12头文件给Python3.11编译。首条成功环境使用Python3.11.13。系统有对应python-devel则直接使用；否则请由环境提供者安装匹配开发文件。服务器此前使用官方CPython3.11.13源码Include加同一系统Python的pyconfig.h，并设置CPATH/C_INCLUDE_PATH指向该目录。

示例环境变量（替换为自己已安装的路径）：

```bash
export TMPDIR=/data/you/tmp
export HF_HOME=/data/you/cache/huggingface
export XDG_CACHE_HOME=/data/you/cache
export TRITON_CACHE_DIR=/data/you/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/data/you/cache/torchinductor
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CPATH=/data/you/python311-headers/include
export C_INCLUDE_PATH=/data/you/python311-headers/include
```

目录需事先创建。下载完成后可设置HF_HUB_OFFLINE=1和TRANSFORMERS_OFFLINE=1；不要在组件仍缺失时启用离线模式。

## Wan环境

成功组合是PyTorch2.6/cu124、SageAttention2.2.0、LightX2V0.5.0、FlashInfer0.2.5+cu124torch2.6、pyzmq27.2.0、comfy-kitchen0.2.15。编译包与Python/CUDA/C++ ABI必须匹配。某些包元数据只显示torch2.6.0，验收记录证明实际运行2.6.0+cu124。请先检查依赖导入和GPU首条，不能依靠pip安装成功推断CUDA兼容。

## Hunyuan与LTX

Hunyuan固定SageAttention＋DeepCache配置；FlashAttention版本需与环境匹配。LTX采用官方distilled多组件流程和CPU卸载；已成功环境torch2.13.0+cu132，不能把Wan环境直接复用给LTX。

## 许可证

此包不重新发布官方模型代码或权重。官方组件各自遵循原仓库和权重许可证。研究Prompt和协作脚本尚未指定对外再分发许可证，先使用私有仓库协作，不擅自添加开源许可证。
