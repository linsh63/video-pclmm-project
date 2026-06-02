# Video PCLMM Project

This project reproduces and extends the PCLMM Chinese video Patronizing and Condescending Language detection workflow as an independent video detection project.

The intended pipeline covers:

- data preparation
- video feature extraction
- audio feature extraction
- text feature extraction
- facial expression feature extraction
- multimodal fusion training and evaluation
- experiment reports, figures, and bad case analysis

Official reference project:

- PCLMM GitHub: https://github.com/dut-laowang/PCLMM
- PCLMM dataset: https://zenodo.org/records/15128981
- Paper: Towards Patronizing and Condescending Language in Chinese Videos: A Multimodal Dataset and Detector, ICASSP 2025

Project notes:

- Keep official code snapshots under `third_party/PCLMM/`.
- Keep project source code under `src/`.
- Keep executable project wrappers under `scripts/`.
- Do not commit raw videos, archives, feature caches, pretrained weights, checkpoints, or private planning notes under `doc/brain/`.

## Environment

Use conda for this project, but create the environment manually instead of using `environment.yml`.

The root filesystem on shared servers can be small, so keep the conda environment, package cache, and temporary install files on `/data4`.

```bash
cd /data2/songxinshuai/linsihan/video-pclmm-project

mkdir -p /data4/songxinshuai/conda/envs
mkdir -p /data4/songxinshuai/conda/pkgs
mkdir -p /data4/songxinshuai/tmp

CONDA_PKGS_DIRS=/data4/songxinshuai/conda/pkgs \
conda create -y -p /data4/songxinshuai/conda/envs/video-pclmm \
  --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r \
  python=3.10 pip

conda activate /data4/songxinshuai/conda/envs/video-pclmm

python -m pip install --upgrade pip setuptools wheel

TMPDIR=/data4/songxinshuai/tmp \
PIP_NO_CACHE_DIR=1 \
python -m pip install --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

CONDA_PKGS_DIRS=/data4/songxinshuai/conda/pkgs \
conda install -y -p /data4/songxinshuai/conda/envs/video-pclmm \
  --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  ffmpeg

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
ffmpeg -version
```

Run these commands manually on the target machine, because they install packages or inspect the local GPU/driver environment.

If conda fails with `SSLEOFError` while reading `repodata.json.zst`, prefer the `--override-channels` mirror commands above instead of retrying the default channels.

`openai-whisper==20240930` currently needs `--no-build-isolation` in this environment because its build step imports `pkg_resources`. The project `requirements.txt` intentionally omits `whisper==1.1.10`; the official script needs OpenAI Whisper's `whisper.load_model`, and the separate `whisper` package can conflict with that import.

`Pillow` is pinned to `10.2.0` in the project requirements because `facenet_pytorch==2.6.0` requires `Pillow>=10.2.0,<10.3.0`.
