from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
from scipy.fftpack import dct
from scipy.io import wavfile
from tqdm import tqdm


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
AUDIO_FOLDER_ROOT = Path(os.environ.get("AUDIO_FOLDER_ROOT", RUNTIME_ROOT / "temp" / "WAV"))
FEATURE_FOLDER_ROOT = Path(os.environ.get("FEATURE_FOLDER_ROOT", RUNTIME_ROOT / "features" / "AUDIO_features"))


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595 * np.log10(1 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700 * (10 ** (np.asarray(mel) / 2595.0) - 1)


def load_wav(path: Path) -> tuple[int, np.ndarray]:
    sr, audio = wavfile.read(path)
    audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        audio /= np.iinfo(audio.dtype).max
    max_abs = np.max(np.abs(audio)) if audio.size else 0
    if max_abs > 1:
        audio /= max_abs
    return sr, audio


def frame_signal(audio: np.ndarray, sr: int, frame_size: float = 0.025, frame_stride: float = 0.01) -> np.ndarray:
    frame_length = int(round(frame_size * sr))
    frame_step = int(round(frame_stride * sr))
    signal_length = len(audio)
    if signal_length == 0:
        return np.zeros((1, frame_length), dtype=np.float32)

    num_frames = int(np.ceil(float(abs(signal_length - frame_length)) / frame_step)) + 1
    pad_signal_length = num_frames * frame_step + frame_length
    pad_signal = np.append(audio, np.zeros((pad_signal_length - signal_length,), dtype=np.float32))

    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1))
        + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    )
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)
    return frames


def mel_filterbank(sr: int, nfft: int, nfilt: int = 40) -> np.ndarray:
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)
    mel_points = np.linspace(low_mel, high_mel, nfilt + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((nfft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((nfilt, nfft // 2 + 1))
    for j in range(nfilt):
        left, center, right = bins[j], bins[j + 1], bins[j + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for i in range(left, center):
            fbank[j, i] = (i - left) / (center - left)
        for i in range(center, min(right, nfft // 2 + 1)):
            fbank[j, i] = (right - i) / (right - center)
    return fbank


def extract_mfcc(path: Path, num_ceps: int = 40, nfilt: int = 40, nfft: int = 512) -> np.ndarray:
    sr, audio = load_wav(path)
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1]) if audio.size > 1 else audio
    frames = frame_signal(audio, sr)

    mag_frames = np.absolute(np.fft.rfft(frames, nfft))
    pow_frames = (1.0 / nfft) * (mag_frames**2)

    fbank = mel_filterbank(sr, nfft, nfilt)
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    log_banks = np.log(filter_banks)
    mfcc = dct(log_banks, type=2, axis=1, norm="ortho")[:, :num_ceps]
    return np.mean(mfcc, axis=0).astype(np.float32)


def main() -> None:
    FEATURE_FOLDER_ROOT.mkdir(parents=True, exist_ok=True)

    for subfolder in os.listdir(AUDIO_FOLDER_ROOT):
        subfolder_path = AUDIO_FOLDER_ROOT / subfolder
        if not subfolder_path.is_dir():
            continue

        feature_subfolder = FEATURE_FOLDER_ROOT / subfolder
        feature_subfolder.mkdir(parents=True, exist_ok=True)
        wav_files = sorted(path for path in subfolder_path.iterdir() if path.suffix == ".wav")

        for wav_file in tqdm(wav_files):
            output_file = feature_subfolder / wav_file.with_suffix(".p").name
            if output_file.exists():
                print(f"特征已经存在，跳过: {output_file}")
                continue

            try:
                mfcc_feature = extract_mfcc(wav_file)
                with output_file.open("wb") as fp:
                    pickle.dump(mfcc_feature, fp)
                print(f"成功提取并保存特征: {output_file}")
            except Exception as exc:
                print(f"处理文件 {wav_file.name} 时出错: {exc}")


if __name__ == "__main__":
    main()
