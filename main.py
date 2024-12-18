# main.py

from fastapi import FastAPI, UploadFile, HTTPException
import os
import uuid
import soundfile as sf

# audio_processing.py から関数をインポート
from audio_processing import remove_noise, transcribe_audio_local

app = FastAPI()

@app.post("/process-audio/")
async def process_audio(file: UploadFile):
    try:
        # 一時保存ディレクトリ
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)

        # アップロードされたファイルを保存
        temp_filename = f"{uuid.uuid4().hex}.wav"
        input_path = os.path.abspath(os.path.join(temp_dir, temp_filename))
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # ファイルが正しく保存されたか確認
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"アップロードされたファイルが見つかりません: {input_path}")

        print(f"ファイルが保存されました: {input_path}")

        # ノイズ除去
        sampling_rate, cleaned_audio = remove_noise(input_path)

        # ノイズ除去後のファイルを保存
        cleaned_filename = "cleaned_" + temp_filename
        cleaned_path = os.path.abspath(os.path.join(temp_dir, cleaned_filename))
        # soundfile を使用してオーディオを保存
        sf.write(cleaned_path, cleaned_audio, sampling_rate, subtype='PCM_16')

        # ノイズ除去後のファイルが存在するか確認
        if not os.path.exists(cleaned_path):
            raise FileNotFoundError(f"ノイズ除去後のファイルが見つかりません: {cleaned_path}")

        print(f"ノイズ除去後のファイルが保存されました: {cleaned_path}")

        # Whisper で文字起こし
        transcription = transcribe_audio_local(cleaned_path)

        # 成功時のレスポンス
        return {"message": "ファイルが正常に処理されました", "transcription": transcription}

    except Exception as e:
        print(f"エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))
