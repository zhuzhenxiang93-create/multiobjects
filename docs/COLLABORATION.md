# 同学生成任务：领取后运行

## 这次可领取的任务

当前快照已排除全部已完成视频，以及学校服务器运行/排队中的清单：

| 模型 | 可领取数量 | 批次ID |
|---|---:|---|
| Wan2.2 | 250 | wan-20261016-01 到 wan-20261016-05，每批50条 |
| Hunyuan | 40 | hunyuan-20261016-01、02，每批20条 |
| LTX-2.5 | 50 | ltx25-20261016-01、02，每批25条 |
| SANA | 0 | 剩余任务已在学校服务器运行 |

这些固定清单中，同一模型不会重复使用同一Prompt ID；不同模型使用同一个Prompt是必要的对照。每批按物体数量从多到少运行。

## 首次准备（每台机器一次）

1. 仓库所有者邀请同学为collaborator；同学接受邀请，并让本机Git具备push权限。使用各自账号，不共享Token或SSH私钥。
2. 各模型需要独立环境、已下载权重和GPU资源。按docs/INSTALL.md和verification/software_versions.json准备对应版本。首次搭环境不能仅靠git pull完成。
3. 查看profiles/local.example.json：默认把环境放在WORKSPACE/envs/MODEL、源码放在WORKSPACE/source/仓库名、权重放在WORKSPACE/models/组件名。
4. 如果机器上的路径不同，复制profiles/local.example.json为profiles/local.json，填自己的路径。该文件已gitignore。

## 平时只需pull后运行

```bash
git clone https://github.com/zhuzhenxiang93-create/multiobjects.git
cd multiobjects
# 已clone的同学：git switch main && git pull --ff-only
python3 scripts/run_assignment.py --list
```

选择一批，使用自己的GitHub用户名和自己的数据目录：

```bash
python3 scripts/run_assignment.py wan-20261016-01 --worker YOUR_GITHUB_USERNAME --workspace /data/YOURNAME/video-generation
```

本机路径不同则使用配置文件：

```bash
python3 scripts/run_assignment.py wan-20261016-01 --worker YOUR_GITHUB_USERNAME --config profiles/local.json
```

可先加`--dry-run`，只打印实际运行命令，不领取、不加载GPU。实际运行须在已分配的GPU作业中进行，不要在登录节点直接生成。

启动顺序：本机环境检查 → 仓库版本检查 → 原子领取 → 批量生成 → 逐条解码验收 → 将领取记录标为complete/failed。

## 如何避免重复

- 每批有唯一ID和不可变清单哈希。
- 领取结果写入共享GitHub的`claims/批次ID`分支，使用带预期旧值的Git原子更新。两个同学同时领取时只有一个会成功。
- 没有仓库写权限或网络检查失败时，拒绝启动生成。
- 同一个GitHub用户名换到另一台机器，也不能自动重复领取；续跑需保留原机器WORKSPACE/collaboration/批次ID/claim.json。
- 同一台机器还有进程文件锁，防止两个终端同时运行。
- 原子领取不会修改main，也不会上传权重或视频。
- 不要绕过入口直接运行run_batch.py或重新切分相同清单；那样不能提供跨机器去重。
- 领取不会自动过期，避免长视频任务还在运行时被别人误领。需要换人时，先确认原任务已停止，由负责人安排移交。
- 学校服务器另有collaborator预留清单。后续学校任务必须排除这些ID，除非负责人明确回收未领取任务。

## 中断、失败与结果回传

在原机器使用相同目录和同一命令重启，会复用原领取记录并跳过已经通过验收的视频。若已经整批complete，入口会停止而非重生成。

新配置、换源码或改依赖可能使续跑指纹不匹配，需要负责人处理；不要自行换Prompt或seed来覆盖原任务。

输出目录：WORKSPACE/results/MODEL/批次ID/PromptID/。每条有video.mp4、result.json、attempt-*/generation.log。
请把这些结果通过约定的共享存储交还负责人；GitHub只保存领取/完成状态，不保存视频，完成状态不代表文件已经传回学校服务器。

## Slurm机器

在自己的sbatch脚本中激活对应环境/设置GPU资源，最后执行上面的run_assignment.py命令。领取在作业真正运行时发生，因此不用在排队阶段占住任务。

## 限制

本入口通过CPU测试与实际本地Git并发测试，四模型适配器通过参数与部署修复测试；新的统一入口未额外提交重复视频做GPU验收。服务器原模型配置已有成功记录，每位同学首批应先观察首条结果。
