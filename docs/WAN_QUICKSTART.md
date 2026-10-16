
## 1. 一次性准备

- 独立Python3.12环境，已验证组合：PyTorch2.6.0+cu124、SageAttention2.2.0、LightX2V0.5.0、FlashInfer0.2.5+cu124torch2.6、transformers4.51.3、diffusers0.39.0、pyzmq27.2.0、comfy-kitchen0.2.15。
- LightX2V源码固定commit：cabb8bb684e44af5aa19be437f2e15082855cb96。
- 权重必须是Wan2.2-T2V-A14B官方原生权重，不是TI2V-5B或Diffusers格式。
- Git有本仓库push权限，ffmpeg/ffprobe可用，在分配到GPU的环境运行。
- 环境安装步骤与CUDA/ABI说明见INSTALL.md；没有一键安装承诺，首次仍需搭好环境和下载权重。

已在H200上验证720p首条与200条批量；别的GPU需要实际验收。首条成功不能保证所有输入都无异常，已知全零V测试仍异常。

## 2. 只填Wan的三个路径

```bash
git pull --ff-only
cp profiles/wan.example.json profiles/local.json
```

填写profiles/local.json：

- python：Wan环境中的Python可执行文件。
- source：上述固定版本的LightX2V源码目录。
- weights：Wan2.2-T2V-A14B原生权重目录。
- workspace：缓存、日志和结果所在的数据盘目录。

模板含VIDEO_WORKSPACE变量；可以把路径直接改成绝对路径，或运行时传--workspace。个人配置不会上传到Git。

## 3. 领取一批并运行

```bash
python3 scripts/run_assignment.py wan-20261016-01 --worker YOUR_GITHUB_USERNAME --config profiles/local.json --workspace /data/YOURNAME/video-generation
```

可先加--dry-run查看命令。预检通过后才领取，不会为了这个任务加载其他模型。

Wan共5批，按需要挑一个：

| 批次 | 条数 |
|---|---:|
| wan-20261016-01 | 50 |
| wan-20261016-02 | 50 |
| wan-20261016-03 | 50 |
| wan-20261016-04 | 50 |
| wan-20261016-05 | 50 |

每批已排除学校服务器正在运行、排队和已完成的Wan任务；数量按多到少排列。别人已经领取时会拒绝运行，不要绕过领取入口。

## 4. 固定生成配置和结果

1280×720、81帧、16FPS、40步、seed42、SageAttention2。不要自行修改Prompt、seed、分辨率或步数。

输出在workspace/results/wan/批次ID/。中断后在同一机器、同一数据目录执行相同命令续跑。完成后将video.mp4、result.json及日志交回负责人；视频不通过GitHub上传。

完整领取锁和失败移交说明见COLLABORATION.md。
