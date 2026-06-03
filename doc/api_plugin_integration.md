# 视频审核 API 插件接入指南

## 1. 插件定位

这个项目在小组系统中的角色是一个独立的视频审核服务。

它接收一个 `.mp4` 视频文件，返回该视频是否可以被视为正常视频：

```json
{
  "is_normal": true
}
```

业务系统可以把它当作一个 HTTP 插件调用：

```text
前端上传视频
-> 业务后端暂存视频
-> 业务后端调用本项目 /predict
-> is_normal=true  才正式发布
-> is_normal=false 拒绝发布或进入人工审核
```

生产环境不建议让浏览器前端直接调用本服务。推荐由业务后端调用本服务，这样可以隐藏模型服务地址、避免用户绕过审核，也方便做鉴权、日志和重试。

## 2. 当前能力

当前 API 使用 PCLMM 多模态视频检测流程：

- ViT 提取画面特征
- MFCC 提取音频特征
- Whisper 转写语音
- BERT 提取文本特征
- FER-VT 提取人脸表情特征
- MultiModalCrossAttention 融合判断

标签语义：

```text
score = 模型认为视频属于异常 / PCL 的概率
is_normal = score < threshold
```

当前验证结果基于 `disabled` 子集：

```text
训练集 accuracy: 1.0000
测试集 accuracy: 0.8387, threshold=0.5
测试集 accuracy: 0.8710, threshold≈0.32
```

如果业务优先追求整体准确率，可以先使用：

```text
PCLMM_API_THRESHOLD=0.32
```

如果业务优先减少漏检，可以进一步降低阈值，但误伤正常视频会增加。

## 3. 启动服务

先进入项目目录并激活环境：

```bash
cd /data2/songxinshuai/linsihan/video-pclmm-project
conda activate /data4/songxinshuai/conda/envs/video-pclmm
```

推荐启动方式：

```bash
PCLMM_API_FEATURE_BACKEND=resident \
PCLMM_API_CUDA_VISIBLE_DEVICES=4,5,6,7 \
PCLMM_API_DEVICE=cuda:3 \
PCLMM_API_VIT_DEVICE=cuda:0 \
PCLMM_API_WHISPER_DEVICE=cuda:1 \
PCLMM_API_BERT_DEVICE=cuda:1 \
PCLMM_API_FACE_DEVICE=cuda:2 \
PCLMM_API_EXTRACTION_MODE=parallel \
PCLMM_API_THRESHOLD=0.32 \
PCLMM_API_VIT_BATCH_SIZE=16 \
HOST=0.0.0.0 \
PORT=8000 \
scripts/run_api.sh
```

说明：

- `PCLMM_API_FEATURE_BACKEND=resident` 表示 API 启动时常驻加载模型，推荐使用。
- `PCLMM_API_EXTRACTION_MODE=parallel` 表示并行提取多模态特征。
- `PCLMM_API_CUDA_VISIBLE_DEVICES=4,5,6,7` 后，服务内部的 `cuda:0/1/2/3` 分别对应物理 GPU `4/5/6/7`。
- `PCLMM_API_THRESHOLD=0.32` 是当前 disabled 测试集上 accuracy 较高的阈值。

如果 resident 版本异常，可以回退到旧的 subprocess 后端：

```bash
PCLMM_API_FEATURE_BACKEND=subprocess \
PCLMM_API_CUDA_VISIBLE_DEVICES=4,5,6,7 \
PCLMM_API_EXTRACTION_MODE=parallel \
scripts/run_api.sh
```

## 4. 健康检查

接口：

```text
GET /health
```

命令：

```bash
curl --noproxy '*' http://127.0.0.1:8000/health
```

返回示例：

```json
{
  "ok": true,
  "feature_backend": "resident",
  "extraction_mode": "parallel",
  "predictor_loaded": true,
  "resident_extractor_loaded": true
}
```

FastAPI 也会自动提供接口文档：

```text
http://127.0.0.1:8000/docs
```

## 5. 视频预测接口

接口：

```text
POST /predict
```

请求格式：

```text
multipart/form-data
file: mp4 视频文件，必填
threshold: 可选，覆盖默认阈值
file_id: 可选，指定缓存 ID
debug: 可选，true 时返回 score、耗时、特征路径
force_recompute: 可选，true 时强制重新抽取特征
```

最小请求：

```bash
curl --noproxy '*' -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@data/raw/videos/disabled/disabled90.mp4"
```

最小返回：

```json
{
  "is_normal": true
}
```

调试请求：

```bash
curl --noproxy '*' -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@data/raw/videos/disabled/disabled90.mp4" \
  -F "debug=true" \
  -F "force_recompute=true"
```

调试返回示例：

```json
{
  "is_normal": true,
  "score": 0.00105,
  "threshold": 0.32,
  "prediction": 0,
  "file_id": "disabled90_xxxxxxxx",
  "feature_backend": "resident",
  "extraction_mode": "parallel",
  "timings": {
    "extract_video_vit.py": 4.48,
    "extract_audio_text.py": 3.89,
    "extract_face_fervt.py": 7.28,
    "fusion_predict": 0.09,
    "request_total": 7.38
  }
}
```

## 6. 业务后端接入示例

推荐由业务后端调用本服务。下面是 Node.js 示例：

```js
import fs from "node:fs";
import FormData from "form-data";
import fetch from "node-fetch";

export async function checkVideoByModel(videoPath) {
  const form = new FormData();
  form.append("file", fs.createReadStream(videoPath));

  const response = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    throw new Error(`Video model API failed: ${response.status}`);
  }

  return await response.json();
}
```

业务上传流程建议：

```text
1. 前端上传视频到业务后端
2. 业务后端保存到临时目录
3. 业务后端调用本项目 /predict
4. is_normal=true: 移动视频到正式存储，创建帖子/动态
5. is_normal=false: 删除或隔离临时视频，返回审核不通过
```

前端状态可以设计为：

```text
uploading   正在上传
reviewing   后端正在审核
approved    审核通过并发布
rejected    审核失败
failed      技术错误
```

## 7. 前端开发调试示例

如果只是本地开发调试，浏览器也可以直接上传到模型 API：

```js
async function predictVideo(file) {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    throw new Error("video review failed");
  }

  return await response.json();
}
```

生产环境不要依赖这种直连方式，因为用户可以绕过前端逻辑。

## 8. 错误处理

常见状态：

```text
200: 成功返回审核结果
400: 上传文件不是 mp4
500: 模型推理、特征抽取或环境依赖出错
```

业务后端建议：

- 对 `500` 做失败重试或进入人工审核。
- 不要把模型内部错误栈直接暴露给最终用户。
- 对上传文件大小和视频时长做业务限制。
- 对请求加超时控制，短视频可按 30~60 秒设置，长视频建议异步审核。

## 9. 性能说明

在当前 8 卡 3090 服务器上，`disabled90.mp4` 约 10 秒视频的冷缓存耗时：

```text
串行 subprocess: 约 65 秒
并行 subprocess: 约 47 秒
resident 并行:    约 7.4 秒
```

所以推荐：

```text
PCLMM_API_FEATURE_BACKEND=resident
PCLMM_API_EXTRACTION_MODE=parallel
```

当前主要瓶颈一般是：

```text
Face / FER-VT
ViT
Whisper
```

如果后续还要压低延迟，可以考虑：

- 减少 Face 抽帧数
- 减少 ViT 抽帧数
- 增大 ViT batch size
- 换更小的 Whisper 模型
- 按视频时长做异步审核

## 10. 需要准备但不提交的文件

public 仓库不会提交这些大文件：

```text
pretrained/
outputs/checkpoints/
data/raw/
features/
outputs/api/
```

部署机器需要自行准备：

```text
outputs/checkpoints/multi_modal_cross_attention_model.pth
pretrained/googlevit-base-patch16-224-in21k/
pretrained/bert_chinese/
pretrained/FER-VT/
Whisper cache，例如 /data4/songxinshuai/cache/whisper/large-v3.pt
```

如果缺少这些文件，API 启动或第一次请求会失败。

## 11. 接入验收清单

小组成员接入时可以按这个清单验收：

- `GET /health` 返回 `ok=true`
- `predictor_loaded=true`
- `resident_extractor_loaded=true`
- 上传一个 `.mp4` 可以返回 `{"is_normal": true/false}`
- `debug=true` 时可以看到 `score` 和 `timings`
- 业务后端只在 `is_normal=true` 时正式发布视频
- 异常情况不会直接发布视频

