import requests
from docx import Document
import os
from datetime import datetime

CHUNK_SIZE = 1024
API_KEY = "sk_9cb8fc1fa8d204870d890050a10f6f5e3fc144e1a6b783fd"

url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": API_KEY
}

def read_docx(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ الملف {file_path} غير موجود!")
    
    doc = Document(file_path)
    full_text = [para.text for para in doc.paragraphs]
    return '\n'.join(full_text)

# البحث عن أحدث ملف سكريبت تم إنشاؤه
script_files = [f for f in os.listdir() if f.startswith("voice_over_") and f.endswith(".docx")]
if not script_files:
    raise FileNotFoundError("❌ لم يتم العثور على أي ملف سكريبت!")

latest_script = max(script_files, key=os.path.getctime)

print(f"📄 سيتم تحويل الملف: {latest_script}")

text_from_docx = read_docx(latest_script)

data = {
    "text": text_from_docx,
    "model_id": "eleven_multilingual_v2",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.5
    }
}

try:
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()

    # إنشاء اسم ملف صوتي يحتوي على التاريخ والوقت
    audio_filename = datetime.now().strftime("voice_over_%Y%m%d_%H%M%S.mp3")

    with open(audio_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    print(f"✅ الصوت تم إنشاؤه وحُفظ باسم {audio_filename}")

except requests.exceptions.RequestException as e:
    print(f"❌ حدث خطأ أثناء الاتصال بـ API: {e}")
