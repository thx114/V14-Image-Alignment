# 2026-11-06 发布验证

本次验证使用独立生成的彩色平滑噪声和编号文字合成图，不改动原 Train/Train2 数据。不是游戏图质量验收，不证明历史候选逐像素复现或严格边界精度。

环境：Windows、Python 3.11+、OpenCV 4.13.0.92、NumPy 2.3.2、SciPy 1.17.1、Pillow 12.2.0、psutil 7.1.3。

实际执行：

- `init` 对2208×1242源 PNG 建立 SHA256 manifest。
- 完整 `prepare`：SIFT、多尺度 flow、定向 SVF、v14 残差、raw-once 候选及两档缩放均实际执行，约18.31秒。
- 主画布新增支持损失0，有限 map，Jacobian 最小2.9898378849；1080p与720p新增支持损失均0，Jacobian最小分别3.3753335476、7.5963721275。
- 实际查看6张原生并列混合 tile，合成编号与纹理未见明显新增双线。仅据此对这份合成 fixture 登记 ACCEPT，不推广成人物边界验收。
- 实际 `review → apply → verify` 成功；输出文件 SHA256、保护源/raw、raw map重放、两档缩放、manifest和锁释放核验通过。
- reference 原始源码哈希全部匹配；AST 对比显示便携 pipeline 的函数体只有 `prereg` 改动（policy路径和移除缺失诊断器哈希）。v12 工作函数的拟合与落盘前缀保持原 AST 一致，仅去掉后续独立评估块。

尚未验证：其他输入尺寸、非Windows系统、全量真实数据重新视觉审核、视频闪烁、所有故障回滚分支、严格全图<2px和人物边缘<1.5px。历史 NCC evaluator 不在本次软件中，详见 PROVENANCE。
