from fastapi import FastAPI, UploadFile, HTTPException
import os
import uuid
import soundfile as sf
from audio_processing import remove_noise, transcribe_audio_local

app = FastAPI()

@app.post("/process-audio/")
async def process_audio(file: UploadFile):
    try:
        # 一時保存用ディレクトリ
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)

        # アップロードファイルを保存
        temp_filename = f"{uuid.uuid4().hex}.wav"
        input_path = os.path.abspath(os.path.join(temp_dir, temp_filename))
        with open(input_path, "wb") as f:
            f.write(await file.read())

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"アップロードされたファイルが見つかりません: {input_path}")

        print(f"ファイルが保存されました: {input_path}")

        # (1) ノイズ除去なしでの文字起こし
        transcription_original = transcribe_audio_local(input_path, language='ja')

        # (2) ノイズ除去
        sr, cleaned_audio = remove_noise(input_path)
        cleaned_filename = "cleaned_" + temp_filename
        cleaned_path = os.path.abspath(os.path.join(temp_dir, cleaned_filename))
        sf.write(cleaned_path, cleaned_audio, sr, subtype='PCM_16')

        if not os.path.exists(cleaned_path):
            raise FileNotFoundError(f"ノイズ除去後のファイルが見つかりません: {cleaned_path}")

        print(f"ノイズ除去後のファイルが保存されました: {cleaned_path}")

        # ノイズ除去後での文字起こし
        transcription_cleaned = transcribe_audio_local(cleaned_path, language='ja')

        return {
            "message": "ファイルが正常に処理されました",
            "transcription_original": transcription_original,
            "transcription_cleaned": transcription_cleaned
        }

    except Exception as e:
        print(f"エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))
