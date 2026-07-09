# 模块移植与验证指南（精简版）

本指南仅保留 recon_4dhoi 管线的核心数据流、文件契约与最小验证流程。
适用于替换 Step 1/2/3 中任一模块（如 GEM-SMPL、WiLoR、SAM2、MoGe、FoundationPose）。

## 1. 管线数据流

recon_4dhoi 当前默认执行 5 个阶段（Step 1-5）：

- Step 1：人体运动预测（GEM-SMPL + WiLoR）
- Step 2：预处理（SAM2 masks + MoGe depth）
- Step 3：物体位姿估计（FoundationPose）
- Step 4：4D HOI 优化（HOIOptimizer）
- Step 5：过滤与后处理

数据流为单向文件流：

- video -> Step 1/2/3
- Step 1 输出 -> Step 4
- Step 2 输出 -> Step 3/4
- Step 3 输出 -> Step 4
- Step 4 输出 -> Step 5

结论：模块可独立替换，前提是输出文件格式不变。

## 2. 文件契约

以下路径相对 results/generation：

### 2.1 上游输入

- videos_kling/{dataset}/{category}/{video_id}.mp4
- mesh/{dataset}/{category}/model.obj
- foundation_pose/{dataset}/{category}/{video_id}/first_frame_pose.pickle
- foundation_pose/{dataset}/{category}/{video_id}/cam_K.txt
- foundation_pose/{dataset}/{category}/{video_id}/masks/000000.png
- foundation_pose/{dataset}/{category}/{video_id}/human_masks/000000.png

### 2.2 Step 1 输出（供 Step 4）

- hmr_smplx/{dataset}/{category}/{video_id}.npz
- 结构：
  - motion_global: dict
  - motion_incam: dict
- 关键字段（motion_incam）：
  - predicted_body_height: float
  - foot_contact_probs: (L, 4) 或 None

### 2.3 Step 2 输出（供 Step 3/4）

- 4dhoi_recon_cache/masks/{dataset}/{category}/{video_id}.npz
  - 内容为 masks 字典（frame_idx -> {0: object, 1: human}）
- 4dhoi_recon_cache/depth/{dataset}/{category}/{video_id}.pt

### 2.4 Step 3 输出（供 Step 4）

- foundation_pose_output/{dataset}/{category}/{video_id}/pose_estimation_output/poses_in_cam.pkl
- 格式：List[np.ndarray(4, 4)]，逐帧物体位姿（相机坐标系）

### 2.5 Step 4 输出（供 Step 5）

- 4dhoi_recon_smplx/{dataset}/{category}/{video_id}/hoi_data.pkl

### 2.6 Step 5 输出（最终有效结果）

- 4dhoi_recon_smplx_valid/{dataset}/{category}/{video_id}/hoi_data/hoi_data.pkl

## 3. 移植验证最小流程

### 3.1 基线自检（替换前）

- 使用当前实现跑目标模块
- 与已存在 golden 输出对比
- 记录通过结果（允许使用容差）

### 3.2 模块替换（替换后）

- 保持输出文件路径与结构不变
- 重新运行同一输入样本
- 对比新输出与 golden 输出

### 3.3 判定标准

- 通用结构一致：文件存在、类型一致、shape 一致
- 数值一致：在 rtol/atol 与领域容差内
- 下游可消费：Step n+1 能无改动读取并继续执行

## 4. FoundationPose 验证（最小命令）

### 4.1 运行并对比

python scripts/verify_foundationpose.py \
  --mesh results/generation/mesh/ComAsset/cordless_drill/model.obj \
  --input_dir results/generation/foundation_pose/ComAsset/cordless_drill/kid_indoor2-manipulation_rand00001 \
  --video results/generation/videos_kling/ComAsset/cordless_drill/kid_indoor2-manipulation_rand00001.mp4 \
  --reference_pkl results/generation/foundation_pose_output/ComAsset/cordless_drill/kid_indoor2-manipulation_rand00001/pose_estimation_output/poses_in_cam.pkl

### 4.2 仅对比已有输出

python scripts/verify_foundationpose.py \
  --output_dir /tmp/fp_new \
  --reference_pkl results/generation/foundation_pose_output/ComAsset/cordless_drill/kid_indoor2-manipulation_rand00001/pose_estimation_output/poses_in_cam.pkl \
  --compare_only

## 5. 数据管理约束

- golden 输出一旦选定，不修改内容
- 仅提交核心契约文件，不提交大体积 debug 中间产物
- 推荐单样本先打通，再扩展到多样本

## 6. 相关文件

- scripts/verify_module_output.py
- scripts/verify_step1_wilor.py
- scripts/verify_foundationpose.py
- grail/pipelines/recon_4dhoi.py
- grail/pose_est/object_pose.py
- grail/pose_est/human_pose.py
