# audio_processing.py

import numpy as np
from scipy.fft import fft, ifft
from scipy.io import wavfile
import whisper

# Whisperモデルのロード（モジュールのスコープで一度だけ実行）
model = whisper.load_model("base")  # 必要に応じて "tiny", "small", "large" に変更可能

def remove_noise(file_path):
    """
    音声ファイルのノイズを除去する関数
    :param file_path: 音声ファイルのパス
    :return: サンプリングレートとノイズ除去後のデータ
    """
    # 音声データの読み込み
    sampling_rate, data = wavfile.read(file_path)

    # データ型を浮動小数点数に変換
    data = data.astype(np.float32)

    # フーリエ変換
    freq_data = fft(data)

    # ノイズフィルタ（しきい値設定）
    threshold = np.percentile(np.abs(freq_data), 75)
    freq_data_filtered = np.where(np.abs(freq_data) > threshold, freq_data, 0)

    # 逆フーリエ変換
    cleaned_data = ifft(freq_data_filtered).real

    # 正規化（-1.0～1.0の範囲にスケーリング）
    max_val = np.max(np.abs(cleaned_data))
    if max_val > 0:
        cleaned_data /= max_val

    return sampling_rate, cleaned_data

def transcribe_audio_local(file_path):
    """
    Whisperを使用して音声を文字起こしする関数
    :param file_path: 音声ファイルのパス
    :return: 文字起こしされたテキスト
    """
    try:
        print(f"文字起こし中のファイル: {file_path}")
        result = model.transcribe(file_path, language='ja', fp16=False)
        return result["text"]
    except Exception as e:
        print(f"文字起こし中のエラー: {e}")
        raise e
