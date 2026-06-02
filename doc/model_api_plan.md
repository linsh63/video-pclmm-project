# 底层模型 API 阶段规划

## 目标

本阶段目标是完成一个底层模型 API：

```text
输入：一个本地视频文件或上传视频文件
输出：该视频是否为正常视频的布尔值
```

API 的核心职责是调用训练好的视频检测模型，不负责帖子发布、用户状态、正式存储、业务审核流转等上层产品逻辑。

## 输出形式

优先完成本地可调用版本，再封装 HTTP API。

最终接口语义：

```json
{
  "is_normal": true
}
```

内部可以保留辅助字段用于调试和后续产品扩展：

```json
{
  "is_normal": true,
  "score": 0.18,
  "threshold": 0.5,
  "label": 0
}
```

其中 `score` 表示模型判断为异常 / PCL 视频的概率。若 `score < threshold`，则 `is_normal = true`。

## 阶段边界

本阶段要做：

- 完成训练所需的离线特征抽取。
- 训练一个可加载的融合分类模型。
- 将训练脚本整理成可复用模块。
- 实现单个视频的推理流程。
- 封装一个底层模型 API。

本阶段暂不做：

- 社交媒体完整上传发布系统。
- 用户、帖子、数据库、正式对象存储。
- 通用违规检测能力。
- 多租户、高并发、审计后台。
- 全量产品级内容安全体系。

## 执行步骤

### 1. 完成训练特征准备

目标：让训练脚本可以直接读取四类模态特征。

需要确认以下特征目录齐全：

```text
features/VIT_features/
features/AUDIO_features/
features/TEXT_features/
features/extracted_features_without_xml/
```

每个视频理想上都应有四类特征：

- 画面特征：ViT
- 音频特征：MFCC
- 语音文本特征：Whisper + BERT
- 人脸表情特征：FER-VT

完成标准：

- 已抽取视频数量、音频数量、文本数量、人脸数量一致。
- 特征文件名能和 `Annotation.csv` 中的 `File` 字段对应。
- 训练集和测试集划分能从 `Subset` 字段读取。

### 2. 跑通融合模型训练

目标：先用官方融合模型跑出第一个可用 checkpoint。

输入：

```text
四类特征目录
Annotation.csv
```

输出：

```text
outputs/checkpoints/
outputs/metrics/
```

完成标准：

- 训练过程可以完整跑完。
- 测试集可以输出 accuracy、macro F1、precision、recall 等指标。
- 训练好的模型权重可以保存并重新加载。
- 明确标签含义：`0` 是否代表正常视频，`1` 是否代表异常 / PCL 视频。

### 3. 整理模型代码

目标：把官方实验脚本拆成可维护、可调用的项目代码。

建议模块：

```text
src/models/fusion_model.py
src/data/feature_dataset.py
src/training/train_fusion.py
src/evaluation/evaluate_fusion.py
```

完成标准：

- 模型结构可以被训练代码和推理代码共同 import。
- 训练入口不再依赖硬编码路径。
- checkpoint 保存路径、batch size、epoch、threshold 等参数可以配置。
- 训练和评测结果可以稳定复现。

### 4. 实现单视频推理流程

目标：给一个新视频文件，完成从视频到布尔判断的完整流程。

流程：

```text
video.mp4
-> 抽取四类模态特征
-> 加载 fusion checkpoint
-> 输出异常概率 score
-> 根据 threshold 返回 is_normal
```

建议模块：

```text
src/inference/extract_single_video.py
src/inference/predict_video.py
```

完成标准：

- 可以通过命令行对单个视频做预测。
- 单视频推理不依赖整批数据集目录。
- 临时特征、临时 wav、临时 txt 有独立缓存目录。
- 同一个视频重复推理时可以复用缓存，避免重复抽取特征。

示例命令目标：

```bash
python -m src.inference.predict_video \
  --video path/to/video.mp4 \
  --checkpoint outputs/checkpoints/best_model.pth \
  --threshold 0.5
```

示例输出目标：

```json
{
  "is_normal": true,
  "score": 0.18,
  "threshold": 0.5
}
```

### 5. 封装底层模型 API

目标：提供一个可以被后端业务系统调用的 HTTP API。

建议使用 FastAPI。

接口草案：

```text
POST /predict
```

输入：

```text
multipart/form-data
file: video.mp4
```

输出：

```json
{
  "is_normal": true
}
```

内部流程：

```text
接收上传视频
-> 保存到临时目录
-> 调用单视频推理流程
-> 返回布尔结果
-> 清理或保留调试缓存
```

完成标准：

- API 启动后可以接收一个视频文件。
- API 可以调用训练好的模型完成判断。
- API 返回稳定的 JSON。
- 推理失败时返回明确错误，而不是让服务崩溃。

## 推荐推进顺序

1. 完成当前 `disabled` 子集的四类特征抽取。
2. 用 `disabled` 子集跑通融合模型训练，得到第一个 checkpoint。
3. 整理融合模型代码，去掉硬编码路径。
4. 写单视频命令行推理入口。
5. 用若干已有视频做单视频预测测试。
6. 封装 FastAPI 的 `/predict`。
7. 再考虑扩展到更多数据子集和更稳定的模型指标。

## 当前阶段完成标准

本阶段完成时，项目应满足：

- 有一个训练好的模型 checkpoint。
- 有一个命令行单视频推理入口。
- 有一个 HTTP `/predict` API。
- 输入一个视频后，可以返回 `is_normal: true/false`。
- API 的判断逻辑和训练标签语义一致。

## 风险点

- 当前官方代码以复现实验为主，路径和训练逻辑硬编码较多，需要工程化整理。
- 单视频推理会比纯文本审核慢，主要耗时在 Whisper、ViT 和人脸特征抽取。
- 当前任务更接近中文视频 PCL / 异常语言检测，不等价于通用社交媒体违规检测。
- 如果训练数据只使用部分子集，模型泛化能力会有限，需要后续补充更多数据和评测。
- `threshold=0.5` 只是初始值，后续应根据验证集指标调整。

