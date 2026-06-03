# 视频 PCLMM 检测项目

本项目是一个面向小组系统的视频审核 API 模块。它基于 PCLMM 中文视频多模态检测流程，接收一个 `.mp4` 视频，判断该视频是否为正常视频，并返回布尔值。

当前最重要的交付物是：

```text
视频审核 HTTP API
```

对外核心接口：

```json
{"is_normal": true}
```

## 项目概况

本项目参考 PCLMM 论文和官方代码，完成中文视频 Patronizing and Condescending Language 检测流程的工程化整理。

当前模型使用的模态包括：

- 画面特征：ViT
- 音频特征：MFCC
- 语音文本：Whisper
- 文本语义特征：BERT
- 人脸表情特征：FER-VT
- 融合分类器：MultiModalCrossAttention

标签语义：

```text
score = 模型判断视频属于异常 / PCL 的概率
is_normal = score < threshold
```

当前 `disabled` 子集上的评估结果：

```text
训练集 accuracy: 1.0000
测试集 accuracy: 0.8387, threshold=0.5
测试集 accuracy: 0.8710, threshold≈0.32
```

因此当前推荐阈值为：

```text
PCLMM_API_THRESHOLD=0.32
```

## API 插件接入

推荐接入方式：

```text
前端上传视频
-> 业务后端暂存视频
-> 业务后端调用本项目 /predict
-> is_normal=true  才正式发布
-> is_normal=false 拒绝发布或进入人工审核
```

生产环境不建议让浏览器前端直接调用本服务。更稳妥的方式是由业务后端调用本服务，避免用户绕过审核逻辑，也方便统一做鉴权、日志、重试和人工审核。

详细接入文档：

- [视频审核 API 插件接入指南](doc/api_plugin_integration.md)
- [底层模型 API 阶段规划](doc/model_api_plan.md)

## 启动 API 服务

先进入项目目录并激活环境：

```bash
cd <PROJECT_ROOT>
conda activate <CONDA_ENV_PATH>
```

推荐启动方式：

```bash
PCLMM_API_FEATURE_BACKEND=resident \
PCLMM_API_CUDA_VISIBLE_DEVICES=0,1,2,3 \
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

- `PCLMM_API_FEATURE_BACKEND=resident`：API 启动时常驻加载 ViT、Whisper、BERT、FER-VT 和融合模型。
- `PCLMM_API_EXTRACTION_MODE=parallel`：并行提取多模态特征。
- `PCLMM_API_CUDA_VISIBLE_DEVICES=0,1,2,3` 后，服务内部的 `cuda:0/1/2/3` 对应这里列出的 4 张可见 GPU。
- `PCLMM_API_THRESHOLD=0.32` 是当前测试集中准确率较高的阈值。

如果 resident 后端异常，可以临时回退：

```bash
PCLMM_API_FEATURE_BACKEND=subprocess \
PCLMM_API_CUDA_VISIBLE_DEVICES=0,1,2,3 \
PCLMM_API_EXTRACTION_MODE=parallel \
scripts/run_api.sh
```

## 快速测试

健康检查：

```bash
curl --noproxy '*' http://<VIDEO_API_HOST>:8000/health
```

视频审核：

```bash
curl --noproxy '*' -X POST "http://<VIDEO_API_HOST>:8000/predict" \
  -F "file=@path/to/video.mp4"
```

返回：

```json
{"is_normal": true}
```

调试模式：

```bash
curl --noproxy '*' -X POST "http://<VIDEO_API_HOST>:8000/predict" \
  -F "file=@path/to/video.mp4" \
  -F "debug=true" \
  -F "force_recompute=true"
```

调试模式会额外返回 `score`、`threshold`、`timings`、`feature_paths` 等字段，方便定位延迟和模型判断结果。

## 业务后端调用示例

Node.js 示例：

```js
import fs from "node:fs";
import FormData from "form-data";
import fetch from "node-fetch";

export async function checkVideoByModel(videoPath) {
  const form = new FormData();
  form.append("file", fs.createReadStream(videoPath));

  const apiBaseUrl = process.env.VIDEO_REVIEW_API_URL ?? "http://localhost:8000";
  const response = await fetch(`${apiBaseUrl}/predict`, {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    throw new Error(`Video model API failed: ${response.status}`);
  }

  return await response.json();
}
```

业务后端只应在：

```json
{"is_normal": true}
```

时正式发布视频。

## 性能概况

在测试服务器上，约 10 秒短视频的冷缓存参考耗时：

```text
串行 subprocess: 约 65 秒
并行 subprocess: 约 47 秒
resident 并行:    约 7.4 秒
```

当前推荐使用：

```text
PCLMM_API_FEATURE_BACKEND=resident
PCLMM_API_EXTRACTION_MODE=parallel
```

目前主要耗时在：

```text
Face / FER-VT
ViT
Whisper
```

如果后续继续优化，可以考虑减少抽帧数、增加 batch size、换更小的 Whisper 模型，或将长视频改成异步审核。

## 项目结构

```text
src/api/              FastAPI 服务
src/inference/        单视频推理、常驻特征提取、特征级推理
src/models/           融合模型结构
src/data/             数据与特征读取
src/evaluation/       评估脚本
src/extraction/       兼容版特征抽取脚本
scripts/              启动、准备和运行脚本
third_party/PCLMM/    官方 PCLMM 参考代码
doc/                  项目文档
```

## 不提交的大文件

这些文件和目录不会进入 public 仓库：

```text
data/raw/
features/
pretrained/
outputs/checkpoints/
outputs/api/
temp/
doc/brain/
```

部署机器需要自行准备：

```text
outputs/checkpoints/multi_modal_cross_attention_model.pth
pretrained/googlevit-base-patch16-224-in21k/
pretrained/bert_chinese/
pretrained/FER-VT/
Whisper cache，例如 <WHISPER_CACHE_DIR>/large-v3.pt
```

如果这些文件缺失，API 启动或第一次请求会失败。

## 环境准备

本项目使用 conda 环境，但不使用 `environment.yml`。共享服务器根目录空间可能较小，建议把 conda 环境、包缓存和临时目录放到空间充足的数据盘。

```bash
cd <PROJECT_ROOT>

mkdir -p <CONDA_ROOT>/envs
mkdir -p <CONDA_ROOT>/pkgs
mkdir -p <TMPDIR>

CONDA_PKGS_DIRS=<CONDA_ROOT>/pkgs \
conda create -y -p <CONDA_ENV_PATH> \
  --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r \
  python=3.10 pip

conda activate <CONDA_ENV_PATH>

python -m pip install --upgrade pip setuptools wheel

TMPDIR=<TMPDIR> \
PIP_NO_CACHE_DIR=1 \
python -m pip install --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

CONDA_PKGS_DIRS=<CONDA_ROOT>/pkgs \
conda install -y -p <CONDA_ENV_PATH> \
  --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  ffmpeg
```

检查环境：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
ffmpeg -version
```

说明：

- 如果 conda 读取 `repodata.json.zst` 时出现 `SSLEOFError`，优先使用上面的镜像和 `--override-channels`。
- `openai-whisper==20240930` 当前需要 `--no-build-isolation`，因为构建步骤会导入 `pkg_resources`。
- 项目依赖中不包含 `whisper==1.1.10`，避免和 OpenAI Whisper 的 `whisper.load_model` 冲突。
- `Pillow` 固定为 `10.2.0`，因为 `facenet_pytorch==2.6.0` 要求 `Pillow>=10.2.0,<10.3.0`。

## 官方参考

- PCLMM GitHub: https://github.com/dut-laowang/PCLMM
- PCLMM Dataset: https://zenodo.org/records/15128981
- Paper: Towards Patronizing and Condescending Language in Chinese Videos: A Multimodal Dataset and Detector, ICASSP 2025
