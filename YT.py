# youtube_automation_app.py
import os
import streamlit as st
import requests
import json
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import TextClip, concatenate_videoclips
from io import BytesIO
from datetime import datetime
import openai

# ============ CONFIG ============
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

# ========= FUNCTIONS ============

def search_trending_topics(niche, max_results=5):
    """Search trending topics in a given niche using YouTube Data API"""
    search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={niche}&type=video&order=viewCount&maxResults={max_results}&key={YOUTUBE_API_KEY}"
    resp = requests.get(search_url)
    data = resp.json()
    results = []
    for item in data.get("items", []):
        title = item["snippet"]["title"]
        video_id = item["id"]["videoId"]
        results.append({"title": title, "video_id": video_id})
    return results

def check_copyright(video_id):
    """Naive copyright check using YouTube captions & description keywords"""
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={YOUTUBE_API_KEY}"
    resp = requests.get(url).json()
    description = resp["items"][0]["snippet"]["description"].lower()
    keywords = ["copyright", "all rights reserved", "licensed"]
    return not any(kw in description for kw in keywords)  # True if safe

def generate_script(topic):
    """Generate video script with OpenAI"""
    prompt = f"Write an engaging, 1-minute YouTube video script about: {topic}"
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return resp.choices[0].message["content"]

def create_thumbnail(text, filename="thumbnail.png"):
    """Create simple thumbnail using Pillow"""
    img = Image.new('RGB', (1280, 720), color=(255, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((50, 300), text, font=font, fill=(255, 255, 255))
    img.save(filename)
    return filename

def create_video(script_text, thumbnail_path, filename="video.mp4"):
    """Create a simple text-based video"""
    clips = []
    lines = script_text.split(". ")
    for line in lines:
        txt_clip = TextClip(line, fontsize=40, color='white', bg_color='black', size=(1280, 720)).set_duration(3)
        clips.append(txt_clip)
    final_clip = concatenate_videoclips(clips)
    final_clip.save_frame(thumbnail_path)  # Use first frame as thumbnail
    final_clip.write_videofile(filename, fps=24)
    return filename

# ========= STREAMLIT APP =========

st.title("📺 YouTube Niche Automation Tool")
st.write("Find trending topics, check copyright, create thumbnails, and generate videos automatically.")

niche = st.text_input("Enter your niche:")
if st.button("Search Trending Topics") and niche:
    topics = search_trending_topics(niche)
    safe_topics = []
    for t in topics:
        if check_copyright(t["video_id"]):
            safe_topics.append(t)
    if safe_topics:
        selected_topic = st.selectbox("Select a topic for video:", [t["title"] for t in safe_topics])
        if st.button("Generate Video Script"):
            script = generate_script(selected_topic)
            st.text_area("Generated Script", script, height=200)
            thumb_path = create_thumbnail(selected_topic)
            st.image(thumb_path, caption="Generated Thumbnail")
            if st.button("Create Video"):
                video_path = create_video(script, thumb_path)
                with open(video_path, "rb") as f:
                    st.download_button("Download Video", f, file_name="youtube_video.mp4")
    else:
        st.warning("No safe topics found.")
