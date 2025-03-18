from typing import List, Optional, Dict, Any
from docx import Document
from datetime import datetime
import requests
import os
import cv2
import pytesseract
import yt_dlp
from ultralytics import YOLO
import google.generativeai as genai
from tqdm import tqdm

# Constants
GENAI_API_KEY = "AIzaSyAJexsERXMnXxVd7w5zBiHqy2TiXwU8Gis"
ELEVENLABS_API_KEY = "sk_9cb8fc1fa8d204870d890050a10f6f5e3fc144e1a6b783fd"
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

# Configure Gemini API
genai.configure(api_key=GENAI_API_KEY)

# Define paths
VIDEO_FOLDER = os.path.join("..", "video")  # Path to the video folder
DOWNLOADED_VIDEOS_FOLDER = os.path.join(VIDEO_FOLDER, "downloaded_videos")
KEYWORDS_FOLDER = os.path.join(VIDEO_FOLDER, "key_words")
SCRIPTS_FOLDER = os.path.join(VIDEO_FOLDER, "scripts")
VOICE_OVER_FOLDER = os.path.join(VIDEO_FOLDER, "voice_over")

# Create folders if they don't exist
os.makedirs(DOWNLOADED_VIDEOS_FOLDER, exist_ok=True)
os.makedirs(KEYWORDS_FOLDER, exist_ok=True)
os.makedirs(SCRIPTS_FOLDER, exist_ok=True)
os.makedirs(VOICE_OVER_FOLDER, exist_ok=True)


def generate_voice_over_script(topic: str, lang: str = "en") -> str:
    """
    Generate a voice-over script using Gemini API.
    """
    fixed_prompt = """
    Write a 60-second voice-over script for a video on the following topic.
    The script should be natural, as if someone is reading it aloud, without timings, musical cues, or additional titles.
    Start the script directly without any title like "Script for YouTube Video".
    Keep it simple, easy to understand, direct, and professional.
    Use short sentences and avoid unnecessary details or extra words.
    """
    final_prompt = fixed_prompt + "\n\nTopic: " + topic + " in " + lang
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(final_prompt)
    text = response.text.strip()

    # Remove unwanted titles
    unwanted_titles = ["Script for YouTube Video", "Voice Over Script"]
    for title in unwanted_titles:
        if text.startswith(title):
            text = text[len(title):].strip()

    return text


def save_script_to_docx(text: str, filename: str) -> None:
    """
    Save the generated script to a Word document in the scripts folder.
    """
    doc = Document()
    doc.add_paragraph(text)
    file_path = os.path.join(SCRIPTS_FOLDER, filename)
    doc.save(file_path)
    print(f"✅ Voice Over Script saved as {file_path}")


def extract_keywords(text: str, main_topic: str) -> List[str]:
    """
    Extract important keywords from the script and ensure each keyword contains the main topic only once.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (f"Extract the most important keywords from the following script and return them as a comma-separated "
              f"list in English. Ensure each keyword includes '{main_topic}' only once:\n\n{text}")

    response = model.generate_content(prompt)
    if response.text:
        # تقسيم النص إلى قائمة كلمات مفتاحية
        keywords = response.text.strip().split(",")

        # تنظيف الكلمات من المسافات الزائدة
        cleaned_keywords = [kw.strip() for kw in keywords]

        # التأكد من عدم تكرار main_topic مرتين داخل أي كلمة مفتاحية
        final_keywords = []
        for kw in cleaned_keywords:
            if main_topic.lower() in kw.lower():
                # إزالة التكرار لو وجد
                kw = kw.replace(main_topic, "").strip()
                kw = f"{main_topic} {kw}".strip()  # ضمان وجود الكلمة المفتاحية الأساسية في البداية
            else:
                kw = f"{main_topic} {kw}".strip()  # لو مش موجودة، نضيفها بطريقة طبيعية

            final_keywords.append(kw)

        return final_keywords

    return []


def save_keywords(keywords: List[str]) -> str:
    """
    Save the extracted keywords to a text file in the key_words folder.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"key_words_{timestamp}.txt"
    file_path = os.path.join(KEYWORDS_FOLDER, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        for keyword in keywords:
            f.write(keyword + "\n")
    print(f"✅ Keywords saved in {file_path}")
    return file_path


from youtube_search import YoutubeSearch
import json
import time

def search_videos(query: str, max_results=10, max_retries=3):
    """
    Search YouTube for videos containing the query in their title.
    """
    for attempt in range(max_retries):
        try:
            # البحث الأول باستخدام العنوان الكامل
            results = YoutubeSearch(query, max_results=max_results).to_json()
            videos = json.loads(results).get("videos", [])

            # لو مفيش نتائج، نبحث باستخدام الكلمة المفتاحية الأساسية فقط
            if not videos:
                print(f"⚠ No exact match found for '{query}', trying a broader search...")
                keywords = query.split()  # تقسيم الكلمات المفتاحية
                if len(keywords) > 1:  # لو فيه أكثر من كلمة
                    broad_query = " ".join(keywords[:2])  # ناخد أول كلمتين
                else:
                    broad_query = keywords[0]

                results = YoutubeSearch(broad_query, max_results=max_results).to_json()
                videos = json.loads(results).get("videos", [])

            return videos
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # انتظار قبل إعادة المحاولة

    return []  # إرجاع قائمة فارغة لو كل المحاولات فشلت


def get_video_duration(duration_str: str) -> float:
    """
    Convert YouTube duration (MM:SS or HH:MM:SS) to seconds.
    """
    if isinstance(duration_str, int):
        return duration_str
    parts = list(map(int, duration_str.split(":")))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return float('inf')


def download_video(video: Dict[str, Any], output_dir: str = DOWNLOADED_VIDEOS_FOLDER) -> str:
    """
    Download the given video using yt-dlp without merging formats.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_url = f"https://www.youtube.com{video['url_suffix']}"
    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'format': 'bestvideo[ext=mp4]',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return os.path.join(output_dir, f"{video['title']}.mp4")


def detect_text_in_video(video_path: str) -> bool:
    """
    Detect text in video frames using OpenCV and Tesseract OCR.
    """
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        if text.strip():
            print("❌ Text detected in video!")
            cap.release()
            return True
    cap.release()
    return False


def detect_logo_in_video(video_path: str) -> bool:
    """
    Detect logos or watermarks using YOLOv5su.
    """
    model_path = "yolov5su.pt"
    yolo_model = YOLO("yolov5su.pt")
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo_model(frame)
        for result in results:
            for box in result.boxes:
                class_name = yolo_model.names[int(box.cls[0])]
                if class_name in ["logo", "watermark", "text"]:
                    print(f"❌ Logo/Watermark detected: {class_name}")
                    cap.release()
                    return True
    cap.release()
    return False


def convert_text_to_speech(text: str, output_file: str) -> None:
    """
    Convert text to speech using ElevenLabs API and save it in the voice_over folder.
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    file_path = os.path.join(VOICE_OVER_FOLDER, output_file)
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print(f"✅ Audio saved as {file_path}")


def main():
    steps = 4  # Total number of main steps
    print("\n🚀 Starting the process...\n")

    # Step 1: Get User Input Before Starting the Progress Bar
    topic = input("\n📌 Enter your script topic : ")

    # Configure the progress bar with custom styling
    progress_bar = tqdm(
        total=steps,
        desc="🔄 Progress",
        colour="green",
        bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} Steps"
    )

    # Step 1: Generate Voice Over Script
    print("\n✍️ Generating Voice Over Script...")
    script_text = generate_voice_over_script(topic)
    script_filename = datetime.now().strftime("voice_over_%Y%m%d_%H%M%S.docx")
    save_script_to_docx(script_text, script_filename)
    progress_bar.update(1)

    # Step 2: Extract Key Sentences
    print("\n📑 Extracting Key Sentences...")
    key_sentences = extract_keywords(script_text, topic)  # Pass the main topic to include it in keywords
    keywords_filename = save_keywords(key_sentences)
    progress_bar.update(1)

    # Step 3: Search and Download Videos
    print("\n🎥 Searching and Downloading Videos...")
    keywords = open(keywords_filename, "r", encoding="utf-8").read().splitlines()
    for keyword in keywords:
        print(f"🔍 Searching for videos with title containing: {keyword}")
        videos = search_videos(keyword)  # Use the keyword as the title for search
        if not videos:
            print(f"No videos found with title containing: {keyword}")
            continue
        for video in videos:
            print(f"⬇ Downloading: {video['title']} ({video['duration']})")
            video_path = download_video(video)
            if detect_text_in_video(video_path) or detect_logo_in_video(video_path):
                print("❌ Video contains text or logos, deleting...")
                os.remove(video_path)
            else:
                print("✅ Video is clean.")
    progress_bar.update(1)

    # Step 4: Convert Script to Speech
    print("\n🔊 Converting Script to Speech...")
    audio_filename = datetime.now().strftime("voice_over_%Y%m%d_%H%M%S.mp3")
    convert_text_to_speech(script_text, audio_filename)
    progress_bar.update(1)

    # Close progress bar
    progress_bar.close()
    print("\n✅ Process completed successfully!\n")


if __name__ == "__main__":
    main()