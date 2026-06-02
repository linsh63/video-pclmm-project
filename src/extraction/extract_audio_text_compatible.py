from __future__ import annotations

import os
from pathlib import Path

import torch
import whisper


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
INPUT_FOLDER_ROOT = Path(os.environ.get("WHISPER_INPUT_ROOT", RUNTIME_ROOT / "temp" / "WAV"))
OUTPUT_FOLDER_ROOT = Path(os.environ.get("WHISPER_OUTPUT_ROOT", RUNTIME_ROOT / "temp" / "TXT"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large")
WHISPER_CACHE_DIR = os.environ.get("WHISPER_CACHE_DIR", "/data4/songxinshuai/cache/whisper")


def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper model={WHISPER_MODEL} on device={device}")
    return whisper.load_model(WHISPER_MODEL, device=device, download_root=WHISPER_CACHE_DIR)


def transcribe_multilingual_audio_and_save(model, audio_path: Path, output_path: Path) -> None:
    result = model.transcribe(str(audio_path), task="transcribe")
    transcribed_text = result["text"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(transcribed_text, encoding="utf-8")
    print(transcribed_text)
    print(f"识别结果已保存到 {output_path}")


def process_all_audios(input_folder_root: Path, output_folder_root: Path) -> None:
    model = load_whisper_model()

    for subfolder in os.listdir(input_folder_root):
        subfolder_path = input_folder_root / subfolder
        if not subfolder_path.is_dir():
            continue

        output_folder = output_folder_root / subfolder
        output_folder.mkdir(parents=True, exist_ok=True)

        for audio_path in sorted(subfolder_path.glob("*.wav")):
            output_path = output_folder / audio_path.with_suffix(".txt").name
            if output_path.exists():
                print(f"Transcription already exists for {audio_path.name}, skipping...")
                continue
            transcribe_multilingual_audio_and_save(model, audio_path, output_path)


if __name__ == "__main__":
    process_all_audios(INPUT_FOLDER_ROOT, OUTPUT_FOLDER_ROOT)
