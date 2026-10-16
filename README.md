**只帮跑 Wan 的同学：请直接阅读 [Wan 独立代码与 README](wan_only/README.md)。**

# 多物体视频生成协作仓库

650条唯一ID Prompt，四个模型的统一批量入口、逐条日志、完整解码验收和按视频续跑。另附同一批200条5–20物体Prompt，按20到5降序排列。

## 同学只帮跑Wan

优先看 **[Wan专用简明说明](docs/WAN_QUICKSTART.md)**，使用 `profiles/wan.example.json` 即可。只准备Wan环境和权重，运行入口不会加载其他模型。

## 已验证的服务器配置

| 模型 | 分辨率 | 帧数 / FPS | 推理方案 | 技术验收 |
|---|---|---|---|---|
| HunyuanVideo-1.5 | 1280×720 | 121 / 24 | 50步，SageAttention2 + DeepCache | 多批生成通过 |
| Wan2.2-T2V-A14B | 1280×720 | 81 / 16 | 40步，LightX2V + SageAttention2 | 23489首条通过 |
| LTX-2.5 | 1280×768 | 193 / 24 | distilled固定8+4步，CPU offload | 23482首条通过 |
| SANA-Video 2.0 5B | 1280×736 | 193 / 24 | 50步，CFG8，shift12，时空分块VAE | 24782首条通过 |

seed默认42；保留原生尺寸，不裁剪或缩放。生成时长与分辨率不同，不能宣称严格同条件比较。
服务器已有成功结果不等于此重构入口已做GPU验收。新入口复用已成功参数与部署修复，目前通过CPU流程测试和四模型命令预检；每位同学仍应先验收一条。

## 文件布局

- `prompts/prompts650.json`：完整输入数据。
- `tasks/shared200_objects20to5.json`：LTX、SANA、Wan正在执行的共同200条；不是待认领清单。
- `tasks/server_reservations.json`：2026-10-16的排除快照（含学校任务和固定同学批次）。
- `configs/`：固定生成与质量检查配置。
- `scripts/run_batch.py`：统一入口。
- `scripts/split_tasks.py`：排除服务器已占用任务，分出互不重叠的同学清单。
- `verification/`：已有技术验收记录和实际包版本。
- `reference/`：服务器原始脚本与源码commit，保留溯源；含服务器固定路径，不直接运行。

## 安装

四个模型使用独立环境，另需Git、ffmpeg、ffprobe。详细版本、权重目录和SANA头文件修复见 [安装说明](docs/INSTALL.md)。不上传模型权重、虚拟环境、视频、Token。

## 同学直接帮跑：推荐入口

已导出尚未生成、未被学校作业占用的340条：**Wan250、Hunyuan40、LTX50**。SANA剩余任务仍在学校运行。任务按数量降序，已切成9批。

```bash
git pull --ff-only
python3 scripts/run_assignment.py --list
python3 scripts/run_assignment.py wan-20261016-01 --worker YOUR_GITHUB_USERNAME --workspace /data/YOURNAME/video-generation
```

**首次需要安装对应GPU环境和权重，或在profiles/local.json填已有路径。**此后无需修改生成代码，pull后使用一条命令即可领取和运行。入口使用GitHub原子领取锁，别人已领取的批次会拒绝运行；需要collaborator写权限。

完整步骤、任务表、中断续跑、结果回传见 **[同学协作说明](docs/COLLABORATION.md)**。

`tasks/collaboration.json`为固定分配快照；实际领取状态看`claims/批次ID`分支。不要直接重新切分或绕过run_assignment.py运行这些同学任务。

## 日志、续跑与返回结果

每条任务产生 `result.json`、`video.mp4` 和 `attempt-*/generation.log`。记录Prompt、seed、配置、源码commit、环境包版本、运行耗时、Slurm ID、失败堆栈和完整解码验收。

成功记录只在实际视频尺寸、帧数、FPS和完整解码通过后写出。再次运行同一命令会检查配置指纹并重新验收完成视频，跳过合格条目；输入或配置变化时要求换目录。失败保留日志并停止批次，重启会从失败视频重新生成。中间采样步骤不恢复。

同一输出目录有本机文件锁。每条任务独立进程，发生CUDA错误不会污染下一次重启；代价是每条重载模型，重构入口的批量速度尚未实测，不能直接套用服务器常驻pipeline耗时。

同学回传视频、result.json和日志到约定共享存储，不通过Git上传大量视频。质量检查只验证文件技术完整性，不判断是否满足Prompt。

## 已知限制

- Wan SageAttention正常随机输入与首条视频通过，但全零V输入诊断仍非有限值，不能宣称所有输入下内核正确。
- SANA的时空分块解码改变解码执行方式，需在论文配置中记录；不等于没有任何近似或边界差异。
- 完整依赖锁尚未导出；版本记录是已成功服务器环境快照，机器CUDA/ABI和编译轮子需匹配。
- GitHub仓库：https://github.com/zhuzhenxiang93-create/multiobjects。公开仓库中的代码与Prompt可被访问；实验视频和权重另行存储。

## CPU流程测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖续跑、失败记录、模型参数适配，并使用实际本地Git远端测试两人并发领取、同名用户换机去重、原机恢复和完成状态。它们不替代GPU首条验收。
