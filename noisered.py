import os
import subprocess
import librosa
import soundfile as sf
import pyloudnorm as pyln
import noisereduce as nr
import numpy as np
from moviepy import VideoFileClip, AudioFileClip
from tqdm import tqdm

INPUT_DIR = r"C:\Users\Bruno\Desktop\noisered\input"
OUTPUT_DIR = r"C:\Users\Bruno\Desktop\noisered\output"
TEMP = os.path.join(OUTPUT_DIR, "_temp")

TARGET_LUFS = -15.5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP, exist_ok=True)


# 1) EXTRAIR ÁUDIO COM FFmpeg (melhor que moviepy p/ WAV PCM)
def extract_audio(video_path, audio_path):
    cmd = f'ffmpeg -y -i "{video_path}" -ar 48000 -ac 1 -c:a pcm_s16le "{audio_path}"'
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# 2) DEMUCS
def run_demucs(wav_path):
    cmd = f'demucs -n mdx_extra_q --two-stems=vocals "{wav_path}"'
    subprocess.run(cmd, shell=True)
    base = os.path.basename(wav_path).replace(".wav", "")
    out = os.path.join("separated", "mdx_extra_q", base, "vocals.wav")
    return out


# 3) NOISEREDUCE
def apply_noisereduce(in_wav, out_wav):
    y, sr = librosa.load(in_wav, sr=None, mono=True)

    # pegar melhor trecho de ruído
    frame = int(sr * 0.5)
    min_energy = 99999
    best = None

    for i in range(0, len(y)-frame, frame):
        chunk = y[i:i+frame]
        e = np.mean(chunk**2)
        if e < min_energy:
            min_energy = e
            best = chunk

    noise_profile = best

    reduced = nr.reduce_noise(
        y=y,
        y_noise=noise_profile,
        sr=sr,
        prop_decrease=0.9,
        freq_mask_smooth_hz=200,
        time_mask_smooth_ms=200,
    )
    
    sf.write(out_wav, reduced, sr)


# 4) SOX → compressão + EQ leve + limiter
def apply_sox(in_wav, out_wav):
    cmd = (
        f'sox "{in_wav}" "{out_wav}" '
        'compand 0.3,1 6:-70,-60,-20 -5 0 '
        'equalizer 100 1.0q -2 '
        'equalizer 300 1.0q 2 '
        'equalizer 4000 1.0q -1 '
        'gain -3'
    )
    subprocess.run(cmd, shell=True)


# 5) NORMALIZAR LUFS
def normalize_lufs(in_wav, out_wav, target_lufs):
    y, sr = librosa.load(in_wav, sr=None)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    norm_audio = pyln.normalize.loudness(y, loudness, target_lufs)
    sf.write(out_wav, norm_audio, sr)


# 6) PROCESSAR VÍDEO
def process_video(video_path, output_path):
    raw = os.path.join(TEMP, "raw.wav")
    nrp = os.path.join(TEMP, "nr.wav")
    proc = os.path.join(TEMP, "proc.wav")
    final = os.path.join(TEMP, "final.wav")
    silent = os.path.join(TEMP, "silent.mp4")

    extract_audio(video_path, raw)

    vocals_path = run_demucs(raw)
    apply_noisereduce(vocals_path, nrp)
    apply_sox(nrp, proc)
    normalize_lufs(proc, final, TARGET_LUFS)

    # 1) Remove áudio original do vídeo
    subprocess.run(
        f'ffmpeg -y -i "{video_path}" -an -c:v copy "{silent}"',
        shell=True
    )

    # 2) Injeta o áudio tratado
    subprocess.run(
        f'ffmpeg -y -i "{silent}" -i "{final}" '
        f'-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 256k -async 1 -af "apad" -shortest "{output_path}"',
        shell=True
    )


videos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".mp4")]

for v in tqdm(videos, desc="Processando vídeos"):
    process_video(os.path.join(INPUT_DIR, v), os.path.join(OUTPUT_DIR, v))

print("🎧 Finalizado com qualidade de estúdio!")
