# 发布来源和移植差异

发布日期：2026-11-06。来源是 NR_models 当前磁盘中的完整 v14 实现，主要入口为 `coordination/train2_v14_1280_20261027/pipeline.py`，而非早期 affine-only adapter。目录日期不是当前执行状态；历史报告也不作为本次视觉验收。

`reference/SHA256.json` 记录本次复制的原始源码哈希，可核对原始快照。reference 中的本地绝对路径只反映历史实现，勿直接执行其批次主函数。便携使用 `v14.py`，不需要原机器的 D 盘布局。

便携移植：

1. 所有执行入口路径改为软件源码目录和显式 `--workspace`；raw 使用 workspace/raw。历史源码中的只读引用路径保留在 reference 中。
2. 复制当前 Train2 pipeline 与 size-parameterised v12；v5 global/flow/render 与 v14 残差拟合数学不改动。不采用早期尚未逐像素等价的 `chain.py` 转写版。
3. 冻结 v5 policy 的旧机器 bound_files 清空，运行时由原 pipeline 对当前代码和图像重新生成绑定。数值参数保持不变。
4. **历史独立 NCC evaluator.py 当前未找到，未伪造实现。** 便携 v12 省略 FIT_RESULT 写完之后的独立诊断评估块；移除 pipeline 对缺失 evaluator 文件的哈希前置。此块位于候选及 map 落盘之后，不参与拟合。不宣称诊断、严格边界或历史像素复现已认证。
5. 便携 CLI 直接调用函数，失败为非零退出，避免原批次入口捕获异常后仍退出0。人工 review 增加必须提供 note 和实际看完全部证据的声明；此声明不自动检测是否真的看图。
6. 原始实现使用 Windows CPU affinity。当前版本不以跨平台可用、任意输入尺寸可用或 GUI 应用发布。

外部依赖：NumPy、OpenCV、SciPy、Pillow、psutil，通过 requirements 安装，不随源码分发二进制依赖。仓库不包含 NVIDIA DLL、训练模型、游戏资产或训练集。本次未替作者自行选定开源许可证；公开可读不等于授予未声明的再许可权利。
