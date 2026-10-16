# 只跑 Wan：同学协作入口

本目录只包含 **Wan2.2-T2V-A14B + LightX2V + SageAttention2** 的代码、固定配置和待跑任务。无需安装其他视频模型。

## 1. 获取代码

```bash
git clone https://github.com/zhuzhenxiang93-create/multiobjects.git
cd multiobjects
git config user.name YOUR_GITHUB_USERNAME
git config user.email YOUR_GITHUB_USERNAME@users.noreply.github.com
```

已有仓库则执行 `git pull --ff-only`。请在 `main` 上运行，Git 需有本仓库写权限且已完成认证；脚本通过独立的 `claims/*` 分支领取任务。

**保持完整 Git checkout。** 可以只阅读和使用 `wan_only/`，但不要把这个子目录单独复制到另一仓库运行，否则无法与其他同学共享领取锁。

## 2. 首次准备环境和权重

已成功服务器的组合如下。这里记录的是实际部署版本；本独立入口还没有在同学的 GPU 上完成首条验收。

| 组件 | 固定版本 |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.6.0+cu124 |
| LightX2V 源码 | `cabb8bb684e44af5aa19be437f2e15082855cb96` |
| SageAttention | 2.2.0 |
| FlashInfer | 0.2.5，匹配 cu124 / torch2.6 的构建 |
| transformers / diffusers | 4.51.3 / 0.39.0 |
| pyzmq / comfy-kitchen | 27.2.0 / 0.2.15 |

以下以自己的数据盘 `/data/YOURNAME/video-generation` 为例：

```bash
export VIDEO_WORKSPACE=/data/YOURNAME/video-generation
mkdir -p "$VIDEO_WORKSPACE"/{envs,source,models,cache,tmp}
export HF_HOME="$VIDEO_WORKSPACE/cache/huggingface"
export XDG_CACHE_HOME="$VIDEO_WORKSPACE/cache"
export TMPDIR="$VIDEO_WORKSPACE/tmp"
python3.12 -m venv "$VIDEO_WORKSPACE/envs/wan"
source "$VIDEO_WORKSPACE/envs/wan/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/ModelTC/LightX2V.git "$VIDEO_WORKSPACE/source/LightX2V"
git -C "$VIDEO_WORKSPACE/source/LightX2V" checkout cabb8bb684e44af5aa19be437f2e15082855cb96
```

然后依据该固定版本 LightX2V 自带的安装文档安装其依赖，保留上表的 PyTorch、CUDA 和扩展版本。SageAttention、FlashInfer 等二进制扩展必须匹配 Python / CUDA / PyTorch ABI；不要把其他机器不匹配的 wheel 直接安装进来。`verification/software_versions.json` 保留软件记录。

权重使用 **官方原生 Wan2.2-T2V-A14B 完整目录**，放在：

```text
/data/YOURNAME/video-generation/models/Wan2.2-T2V-A14B/
```

可以从 [官方权重仓库](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) 下载。需包含 high_noise_model、low_noise_model、文本编码器、tokenizer 和 VAE 等完整组件。不是 TI2V-5B，也不是 Diffusers 转换格式。若已有完整权重，直接使用现有路径。

还需在 PATH 中准备 `git`、`ffmpeg`、`ffprobe`。首次环境准备不是一条运行命令自动完成的；先完成依赖安装与权重准备。

## 3. 填写自己的路径

在 `multiobjects` 仓库根目录：

```bash
cp wan_only/profiles/local.example.json wan_only/profiles/local.json
```

模板默认使用 `VIDEO_WORKSPACE`：

```json
{
  "workspace": "${VIDEO_WORKSPACE}",
  "models": {
    "wan": {
      "python": "${VIDEO_WORKSPACE}/envs/wan/bin/python",
      "source": "${VIDEO_WORKSPACE}/source/LightX2V",
      "weights": "${VIDEO_WORKSPACE}/models/Wan2.2-T2V-A14B"
    }
  }
}
```

也可以直接填写自己的绝对路径。个人配置已被 `.gitignore` 排除，不提交账号、token 或其他密钥。

## 4. 领取一批并运行

先查看批次与运行计划：

```bash
python3 wan_only/scripts/run_assignment.py --list
python3 wan_only/scripts/run_assignment.py wan-20261016-01 \
  --worker YOUR_GITHUB_USERNAME \
  --workspace /data/YOURNAME/video-generation \
  --dry-run
```

在已经分配到 GPU 的终端中正式运行：

```bash
python3 wan_only/scripts/run_assignment.py wan-20261016-01 \
  --worker YOUR_GITHUB_USERNAME \
  --workspace /data/YOURNAME/video-generation
```

五批各50条：`wan-20261016-01` 至 `wan-20261016-05`，共250条，按物体数量从多到少排列。它们是学校任务之外的协作预留批次；实时是否已领取以 GitHub `claims/*` 分支为准。

这个入口会检查源码版本、依赖、GPU 和 SageAttention 小型运算，再原子领取一批。新入口和仓库原来的入口使用**同一批次 ID、同一领取锁**，其他人已领取会停止，不会重复生成。请勿绕过此入口直接调用 `run_batch.py`。

如果使用 Slurm，把上面的正式命令放进自己的作业脚本，并按所在平台的规则申请资源。已验证学校配置为单张 H200；其他卡先验收首条。不要在登录节点运行推理，也不要绕过管理员 hold。

## 5. 固定生成配置

| 项目 | 值 |
|---|---|
| 模型 | Wan2.2-T2V-A14B，LightX2V |
| 分辨率 | 1280×720 |
| 帧数 / 帧率 | 81帧 / 16 FPS，约5.06秒 |
| 采样步数 / seed | 40 / 42 |
| Attention | SageAttention2 |
| CPU 卸载 | T5 开启；主模型及 VAE 关闭 |

配置见 `configs/wan.json` 与 `configs/wan_lightx2v.json`。保持原 Prompt、ID、seed 和配置，不能为了匹配要求数量改变输入。

## 6. 结果、日志和续跑

结果位于：

```text
VIDEO_WORKSPACE/results/wan/wan-20261016-01/
  PROMPT_ID/
    video.mp4
    result.json
    attempt-.../
      request.json
      generation.log
      lightx2v.json
  summary.json
```

- 完整解码、1280×720、81帧和16 FPS 验收通过后才记为 `complete`。
- `result.json` 记录 Prompt、seed、源码版本、软件版本、配置、耗时和错误。
- 某条失败立即停止该批，成功结果保留。
- **同一机器、同一 workspace 执行相同命令即可续跑**；成功文件会复核后跳过，失败项创建新 attempt。不要删除 `workspace/collaboration/批次ID/claim.json`。
- 代码、环境或配置改变后会拒绝复用旧结果，避免混用版本。
- 已全部完成的批次再次执行会提示已完成，不重新生成。
- 更换机器接手前，先确认原进程已停止，并由负责人安排领取锁和结果移交；不要自行删除远程锁。

脚本每条视频会重新启动模型进程，吞吐可能比学校驻留模型批处理慢。文件验收通过只代表视频完整，不代表物体数量符合 Prompt。SageAttention 已有实际视频成功记录，但仍有全零 V 特殊输入产生非有限值的已知问题，不能视为所有输入均已验证。

完成后通过共享盘交回 `video.mp4`、`result.json` 和日志；**不要向 GitHub 上传权重或视频**。
