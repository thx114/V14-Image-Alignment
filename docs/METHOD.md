# 完整 v14 应用方法

## 数据与方向

固定 source，拟合 target 到 source 的对应。保存 map 为 `source_crop_xy -> raw_target_xy`：对每个输出像素给出 raw 中的采样坐标，x 为列，y 为行，整数为像素中心。

先依据明确画布契约创建独立未修正 baseline，用于拟合和前后展示。旧已变形 `output_f` RGB 不作为拟合输入；最终候选从当前 raw 按复合 map 一次 `INTER_LINEAR` 采样。算法内部的灰度拉回/前置 PNG 不构成最终训练 RGB 的多次重采样。

## 1. 全局与多尺度对应

v5 全局：SIFT 12000 特征、ratio 0.72、partial affine RANSAC 2px、10000 次迭代、confidence 0.995、至少 12 内点。全局拟合排除冻结评估查询中心附近 8px 区域。内点拟合残差不代表当前物体边界已经对齐。

多尺度 DIS：尺度 0.125/0.25/0.5/1，MEDIUM、finestScale=0、spatial propagation；CLAHE clip=2、16×16。前后向 cycle≤1.5px、纹理标准差≥3、残差长度≤24px、局部中值一致性≤2.5px。高斯 sigma24/kernel145、支持距离24→40、边缘渐隐64→96、mass floor=1e-6。每个源观测只保留一份，不重复叠加全局约束。

## 2. 定向 SVF 前置配准

历史模块名含 v12，它只作为完整链路的前置组件。高斯 sigma32/kernel385、mass prior=1e-7、支持64→128；在目标边界64px内约束朝外速度，再以 6 次 scaling-and-squaring 积分，alpha=1。尺寸参数从绑定输入推导，不改变数值参数。

## 3. v14 局部残差

DIS MEDIUM、finest0、patch8/stride3、gradient25、variational5；独立前后向 flow，cycle≤0.5px、长度≤4px、源 Sobel≥4、采样边距2px。

置信度 `known * exp(-cycle²/(2*0.5²))`；sigma8 高斯平滑位移分子与置信度质量，归一化分母加0.05；halo64 的 smoothstep。当前通用配置使用全链路画布作为残差域。以浮点双线性方式复合局部 map，末单元线性外推；最终 raw 支持门禁决定是否允许交付，不裁坐标伪造通过。

## 4. raw 复合和输出

将 source→baseline map 乘名义 raw→baseline affine 的逆矩阵，得到 source→raw float32 map。先检查新增支持损失=0、有限性及 Jacobian≥0.05，再从 raw 采样2048×1152候选。1080p和720p各自从同一候选缩小，保存派生几何参考 map，但输出 RGB 的准确重放必须包括 `INTER_AREA`，不能仅用派生 map 直接 remap 后宣称 RGB 一致。

## 5. 视觉与事务

实际查看固定原图+未修正目标、固定原图+完整候选各50%的原生并列混合图。明显双线/扭曲判可见失败；普通纹理、颜色和软化不单独拒绝；结构增删另列内容问题。未看过、支持域未知和入口错误均不是处理后可见失败。

绑定 input/raw/candidate/map/evidence 哈希，审核后才提交。使用一个写入者、排他锁、备份、CAS、PLAN/JOURNAL 和回滚；提交后核对输出、manifest、源保护及 map重放。不得把程序门禁替代实际视觉审核或严格边界认证。
