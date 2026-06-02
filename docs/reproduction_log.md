# Reproduction Log

## Initialization

- Project initialized as an independent PCLMM video detection workspace.
- Git repository initialized.
- Project directory skeleton created.
- Official code added under `third_party/PCLMM/`.
- Official PCLMM source: https://github.com/dut-laowang/PCLMM
- Official PCLMM commit: `9b0745ca06de9de59ac12e8016d2c37a4d5ea7e4`
- Private planning notes under `doc/brain/` are ignored by Git.

## Environment Notes

- Environment management choice: conda.
- Environment setup uses manual `conda create` and `pip install -r requirements.txt`; `environment.yml` is intentionally not used.
- Current base Python observed before creating the project environment: Python 3.12.2.
- Official dependencies are pinned in `third_party/PCLMM/requirements.txt` and mirrored in project `requirements.txt`.
- Commands that create environments, install packages, or modify system dependencies should be executed manually by the project owner.
- Root filesystem has limited free space, so environment creation should use `-p /data4/songxinshuai/conda/envs/video-pclmm`, `CONDA_PKGS_DIRS=/data4/songxinshuai/conda/pkgs`, `TMPDIR=/data4/songxinshuai/tmp`, and `PIP_NO_CACHE_DIR=1`.
- Default conda channels may fail with `SSLEOFError`; use one-off `--override-channels` mirror commands instead of changing global conda config.
- Project `requirements.txt` uses `openai-whisper==20240930` and omits official `whisper==1.1.10` because the official ASR script needs OpenAI Whisper's `whisper.load_model`.
- Install project requirements with `--no-build-isolation` if `openai-whisper` fails with `ModuleNotFoundError: No module named 'pkg_resources'`.
- Project `requirements.txt` uses `Pillow==10.2.0` instead of the official `Pillow==11.1.0` because `facenet_pytorch==2.6.0` requires `Pillow>=10.2.0,<10.3.0`.
