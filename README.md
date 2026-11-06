# V14 Image Alignment

完整 v14 图像配准与安全应用工具，用于将重绘目标对齐到固定原图。CPU Python 命令行软件，包含实际使用过的完整链路源码、原始实现快照、方法说明及可迁移的应用入口。

**链路：SIFT 全局配准 → 多尺度 DIS 对应 → 定向 SVF 前置配准 → v14 局部残差 → 复合到 raw 的 map → raw RGB 一次采样。**

历史名为 v12 的模块仅是前置组件，不能单独作为完整 v14 结果交付。本工具不使用 affine-only adapter，不提供 fallback，不变形原图、不用原图填补目标，不通过模糊掩盖轮廓重影。

## 安装

使用 Python 3.11 或更新版本。以下版本来自本次验证环境；首次安装需要联网下载依赖，之后处理完全本地执行，不需要模型权重、API 或 GPU。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe v14.py --help
```

后文 `python` 指安装好依赖的解释器。Windows 为本次验证平台；原前置组件使用 `psutil.cpu_affinity`，其他系统尚未验证。

## 输入和坐标契约

本次发布保留当前 Train2 的固定配置，不是任意尺寸自动对齐工具。工作目录必须与软件源码目录分开。

```text
my-dataset/
  Train2/input/1.png     固定原图，2208×1242
  raw/1.png              对应重绘原始目标，3840×2160
```

文件按相同数字 ID 配对。源图裁剪 `[80,45,2128,1197]` 得到 2048×1152 链路画布；该画布的 raw 候选只采样一次。1920×1080 和 1280×720 分别由同一候选 `INTER_AREA` 缩小，720p 不从 1080p 二次缩小。

raw 奇数高度且首行符合既定重复行检测时会去掉一行，并记录证据。兼容的 2158 行目标保留既有非等比名义映射标记，必须核对构图，不能把名义映射当作正确对应的证明。其他输入尺寸直接拒绝。

## 应用步骤

```powershell
python v14.py --workspace D:\datasets\my-dataset init
python v14.py --workspace D:\datasets\my-dataset prepare --ids 1
```

`init` 将源图 SHA256 绑定到新 manifest，禁止覆盖已有 manifest。`prepare` 仅准备独立候选、map、回执和原生 1:1 的左右并列混合证据，不提交训练结果。

打开 `artifacts/train2_v14_1280_20261027/items/1/RECEIPT.json` 列出的 **全部 evidence**。每张图左侧为固定原图与未修正目标的 50% 混合，右侧为同一原图与完整候选的 50% 混合。人物、耳部、手脚与地面、桌地等交界需按原生尺度看完；跨 tile 的物体另补连续 ROI，不缩小图片来判断细微重影。

只有实际查看全部证据后才能登记审核。例如：

```powershell
python v14.py --workspace D:\datasets\my-dataset review --ids 1 --decision ACCEPT --viewed-all-evidence --note "审核人、时间、实际查看文件及hash、ROI、具体观察、覆盖边界；strict=UNPROVEN"
python v14.py --workspace D:\datasets\my-dataset apply --ids 1
python v14.py --workspace D:\datasets\my-dataset verify --ids 1
python v14.py --workspace D:\datasets\my-dataset summary
```

实际看过但仍有明显人物/物体交界双线或扭曲时，用 `--decision HOLD`。未看图保持 `NOT_VIEWED`；`--viewed-all-evidence` 是人工声明，不是自动视觉检测器。不要复制上述 ACCEPT 示例作为无人审核的批处理。

提交后结果在 `Train2/input_f`、`output_f`（1080p）、`input_f_1280`、`output_f_1280`（720p）及 `Train2/output`（归一化 raw）中。原始 `Train2/input` 和 `raw` 不覆盖。map、备份、事务日志和审核回执保留在 artifacts 中。每次最多 20 个 ID；出错立即非零退出，排查后再继续。

## 门禁与限制

- 新增支持损失必须为 0；保存 map 有限，所有单元角 Jacobian 最小值 ≥0.05。
- 候选必须能由保存 map 对当前 raw 逐像素重放；两档输出必须精确重放缩放步骤。
- 先实际看图，再 ACCEPT，再提交；应用使用共享排他锁、独立备份、版本比对和事务日志。
- 支持域失败不是视觉失败；不可裁 map、填补缺失区域、降低门槛或回退到旧版本。
- 严格全图 <2px／人物边缘 <1.5px **仍为 UNPROVEN**；Jacobian、支持域和 RGB 重放不能认证语义边界。
- 突然断电、并发冲突与全部回滚分支未做完整故障注入验证。发现 PLAN/JOURNAL 中断应先核对每个文件 old/new hash 和备份，禁止删锁盲重跑。
- 不自动训练，不承诺视频时序稳定，也不保证解决模型运行时闪烁。

详见 [完整方法](docs/METHOD.md)、[发布来源与差异](docs/PROVENANCE.md)、[验证记录](docs/VALIDATION.md)。

## 源码布局

`v14.py` 是便携入口；`portable/` 是当前实际批次实现的路径适配；`reference/` 保留原始完整实现和 SHA256；`config/` 保存冻结 v5 参数。未上传训练集、生成图片、模型权重、账号信息或本地运行回执。

This repository provides the full V14 CPU alignment workflow, native-scale overlay review, raw-once resampling and transactional application. The current profile is fixed to a 2208×1242 source and a roughly 3840×2160 target, producing 1080p and 720p pairs. Strict semantic boundary accuracy remains unproven. See the Chinese workflow above for the required manual review steps.
