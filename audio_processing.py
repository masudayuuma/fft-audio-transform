import librosa
import numpy as np
import noisereduce as nr
import whisper

# Whisperモデル読み込み（必要に応じてモデルサイズ変更: "medium", "large" 等）
model = whisper.load_model("medium")

def remove_noise(file_path):
    """
    スペクトルゲーティングを用いてノイズを除去する関数。
    """
    # オリジナルのサンプリングレートで読み込み
    data, sr = librosa.load(file_path, sr=None)
    reduced_noise = nr.reduce_noise(y=data, sr=sr)
    
    # 正規化
    max_val = np.max(np.abs(reduced_noise))
    if max_val > 0:
        reduced_noise = reduced_noise / max_val
    
    reduced_noise = reduced_noise.astype(np.float32)
    return sr, reduced_noise

def transcribe_audio_local(file_path, language='ja'):
    """
    Whisperを使用して音声を文字起こしする関数
    """
    try:
        print(f"文字起こし中のファイル: {file_path}")
        result = model.transcribe(file_path, language=language, fp16=False)
        return result["text"].strip()
    except Exception as e:
        print(f"文字起こし中のエラー: {e}")
        raise e
