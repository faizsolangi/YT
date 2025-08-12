# app.py
import os
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import requests
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    ImageClip,
    AudioFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
    TextClip,
)
import openai
import googleapiclient.discovery

# -------------------------
# Config & env variables
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")  # optional, for stock images

if not OPENAI_API_KEY:
    st.error("Set OPENAI_API_KEY environment variable.")
    st.stop()
if not YOUTUBE_API_KEY:
    st.error("Set YOUTUBE_API_KEY environment variable.")
    st.stop()

openai.api_key = OPENAI_API_KEY

YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
YOUTUBE_SEARCH_PUBLISHED_AFTER = os.getenv("YOUTUBE_SEARCH_PUBLISHED_AFTER", "2025-01-01T00:00:00Z")

# -------------------------
# Helper utilities
# -------------------------
def safe_file_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in " ._-" else "_" for c in s)[:120]

def fetch_trending_videos_from_youtube(query: str, max_results: int = 10) -> List[dict]:
    youtube = googleapiclient.discovery.build(
        YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY
    )
    req = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        order="viewCount",
        maxResults=max_results,
        publishedAfter=YOUTUBE_SEARCH_PUBLISHED_AFTER,
    )
    res = req.execute()
    results = []
    for item in res.get("items", []):
        vid = item["id"]["videoId"]
        snippet = item["snippet"]
        results.append({
            "video_id": vid,
            "title": snippet.get("title"),
            "channel": snippet.get("channelTitle"),
            "publishedAt": snippet.get("publishedAt"),
            "description": snippet.get("description", ""),
            "url": f"https://www.youtube.com/watch?v={vid}",
        })
    return results

def get_video_license(youtube_client, video_id: str) -> str:
    resp = youtube_client.videos().list(part="status", id=video_id).execute()
    items = resp.get("items", [])
    if not items:
        return "unknown"
    status = items[0].get("status", {})
    return status.get("license", "unknown")  # 'creativeCommon' or 'youtube' (standard)

def call_openai_rank_topics(videos: List[dict]) -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    video_text = "\n".join([f"{i+1}. {v['title']} — {v['url']} — by {v['channel']}" for i, v in enumerate(videos)])
    system = "You are a concise YouTube trend analyst. Output exactly three suggested topic headlines (not full scripts) with 1-line reason each."
    user = f"Here are trending videos in the niche:\n\n{video_text}\n\nReturn in this format:\n1) Title: <title>\n   Reason: <one short reason>\n2) Title: ...\n3) Title: ..."
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"system","content":system}, {"role":"user","content":user}],
        temperature=0.35,
        max_tokens=500
    )
    return resp["choices"][0]["message"]["content"].strip()

def parse_ranked_text(text: str) -> List[Tuple[str,str]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    results = []
    current = {}
    for ln in lines:
        if ln[0].isdigit() and "Title:" in ln:
            title = ln.split("Title:",1)[1].strip()
            current = {"title": title, "reason": ""}
            results.append(current)
        elif "Reason:" in ln and current is not None:
            current["reason"] = ln.split("Reason:",1)[1].strip()
    return [(r["title"], r["reason"]) for r in results]

def llm_copyright_check(topic_title: str, context_text: str) -> Tuple[str,str]:
    """
    Ask LLM to classify risk: returns ("SAFE" or "RISK", explanation)
    """
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system = "You are a cautious content-safety assistant. Classify whether creating a video on the given topic title risks copyright/trademark/rights-of-publicity or uses copyrighted characters/music/brands. Return exactly SAFE or RISK followed by a short reason."
    user = f"Topic title: {topic_title}\n\nContext (examples / related text):\n{context_text}\n\nAnswer format:\nSAFE: short reason OR RISK: short reason"
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"system","content":system}, {"role":"user","content":user}],
        temperature=0.0,
        max_tokens=200
    )
    out = resp["choices"][0]["message"]["content"].strip()
    # Try to parse
    if out.upper().startswith("SAFE"):
        return "SAFE", out.split(":",1)[1].strip() if ":" in out else ""
    elif out.upper().startswith("RISK"):
        return "RISK", out.split(":",1)[1].strip() if ":" in out else ""
    else:
        # fallback heuristic
        return ("RISK", out) if "copyright" in out.lower() or "trademark" in out.lower() else ("SAFE", out)

def generate_script_openai(topic: str) -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system = "You are a professional concise YouTube script writer. Produce a short script suitable for a 60-120s video."
    user = (
        f"Write an engaging YouTube video script for the topic: \"{topic}\".\n"
        "- Hook (first 5-10s)\n- 3 short bullet points for main content\n- Quick summary\n- Call to action: like/subscribe\nKeep sentences short and energetic."
    )
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.6,
        max_tokens=700
    )
    return resp["choices"][0]["message"]["content"].strip()

# -------------------------
# Image helpers (Pexels fallback)
# -------------------------
def pexels_search_images(query: str, per_page: int = 6) -> List[str]:
    if not PEXELS_API_KEY:
        # fallback curated images (safe)
        return [
            "https://images.pexels.com/photos/3183186/pexels-photo-3183186.jpeg",
            "https://images.pexels.com/photos/1181675/pexels-photo-1181675.jpeg",
            "https://images.pexels.com/photos/3861969/pexels-photo-3861969.jpeg",
            "https://images.pexels.com/photos/414519/pexels-photo-414519.jpeg",
            "https://images.pexels.com/photos/842711/pexels-photo-842711.jpeg",
            "https://images.pexels.com/photos/1103539/pexels-photo-1103539.jpeg"
        ][:per_page]
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "per_page": per_page}
    resp = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    items = resp.json().get("photos", [])
    urls = []
    for p in items:
        src = p.get("src", {}).get("landscape") or p.get("src", {}).get("large")
        if src:
            urls.append(src)
    return urls

def download_image(url: str, dest: Path) -> Path:
    r = requests.get(url, stream=True, timeout=20)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return dest

# -------------------------
# TTS (gTTS)
# -------------------------
def text_to_speech_gtts(text: str, out_path: Path, lang: str = "en"):
    tts = gTTS(text, lang=lang)
    tts.save(str(out_path))
    return out_path

# -------------------------
# Video assembly (MoviePy)
# -------------------------
def build_video_from_script_and_images(script_text: str, image_urls: List[str], out_video_path: Path, title_text: str):
    tmpdir = Path(tempfile.mkdtemp())
    # Create audio
    audio_path = tmpdir / "narration.mp3"
    # Use the whole script as narration (for short form). If very long, we could chunk.
    text_to_speech_gtts(script_text, audio_path)
    audio = AudioFileClip(str(audio_path))

    # Download images
    image_files = []
    for i, url in enumerate(image_urls):
        try:
            img_path = tmpdir / f"img_{i}.jpg"
            download_image(url, img_path)
            image_files.append(img_path)
        except Exception as e:
            print("Image download failed:", e)
    if not image_files:
        raise RuntimeError("No images to build video.")

    # make title clip (3s)
    title_clip = TextClip(title_text, fontsize=70, color="white", size=(1280,720), method="caption").set_duration(3).on_color(size=(1280,720), color=(20,20,20))
    # build image clips (each clip duration approx = audio.duration / n_images)
    n = len(image_files)
    per_clip = max(3, audio.duration / max(1, n))  # at least 3s per image
    clips = []
    for img in image_files:
        clip = ImageClip(str(img)).set_duration(per_clip)
        # resize to 1280x720 preserving aspect
        clip = clip.resize(width=1280) if clip.w >= clip.h else clip.resize(height=720)
        # center crop or pad to 1280x720 if needed (MoviePy will letterbox)
        clips.append(clip)

    slideshow = concatenate_videoclips(clips, method="compose")
    # Add title at start
    full = concatenate_videoclips([title_clip, slideshow], method="compose")
    # ensure length >= audio
    if full.duration < audio.duration:
        last = clips[-1].set_duration(audio.duration - full.duration)
        full = concatenate_videoclips([full, last], method="compose")
    final = full.set_audio(audio).set_duration(audio.duration)
    final.write_videofile(str(out_video_path), fps=24, codec="libx264", audio_codec="aac")
    return out_video_path

# -------------------------
# Thumbnail generation (Pillow)
# -------------------------
def generate_thumbnail_pillow(title: str, out_path: Path, subtitle: str = ""):
    W, H = 1280, 720
    bg_color = (18, 18, 18)
    img = Image.new("RGB", (W, H), color=bg_color)
    draw = ImageDraw.Draw(img)

    # fonts: try to load a system font, fallback to default
    try:
        # path to a TTF font may differ by system; adjust if needed
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 64)
        font_sub = ImageFont.truetype("DejaVuSans.ttf", 36)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    # draw a gradient/rectangle accent
    accent = Image.new("RGB", (W, 160), color=(255, 80, 80))
    img.paste(accent, (0, H - 160))

    # Title text (wrap if necessary)
    max_width = W - 120
    # naive wrapping
    words = title.split()
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textsize(test, font=font_title)[0] <= max_width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    y = 140
    for line in lines[:3]:
        tw, th = draw.textsize(line, font=font_title)
        draw.text(((W - tw) / 2, y), line, font=font_title, fill=(255,255,255))
        y += th + 6

    # subtitle
    if subtitle:
        sw, sh = draw.textsize(subtitle, font=font_sub)
        draw.text(((W - sw) / 2, H - 120), subtitle, font=font_sub, fill=(255,255,255))

    img.save(out_path)
    return out_path

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="YouTube Auto Studio (copyright check)", page_icon="🎬", layout="wide")
st.title("YouTube Auto Studio — Niche → Topic → Script → Video → Thumbnail")

with st.sidebar:
    st.header("Config")
    st.write("Ensure env vars:\n- OPENAI_API_KEY\n- YOUTUBE_API_KEY\n(optional) PEXELS_API_KEY")
    st.write("Note: This app builds videos locally and allows downloads. For Render deploy adapt storage for large files.")

col1, col2 = st.columns([3,1])
with col1:
    niche = st.text_input("Enter niche keyword (e.g., 'AI tools', 'solar energy'):", value="AI automation")
with col2:
    max_results = st.number_input("Search results", min_value=3, max_value=20, value=8, step=1)

if st.button("Fetch trending & suggest topics"):
    try:
        with st.spinner("Searching YouTube..."):
            vids = fetch_trending_videos_from_youtube(niche, max_results=int(max_results))
            st.session_state["videos"] = vids
        if not vids:
            st.warning("No videos found.")
        else:
            st.success(f"Fetched {len(vids)} videos.")
            for v in vids[:8]:
                st.markdown(f"- [{v['title']}]({v['url']}) — {v['channel']} — {v['publishedAt']}")
            with st.spinner("Asking OpenAI to pick top 3 topics..."):
                ranked_text = call_openai_rank_topics(vids)
                st.session_state["ranked_text"] = ranked_text
                st.session_state["suggestions"] = parse_ranked_text(ranked_text)
                st.success("Top 3 suggestions generated.")
    except Exception as e:
        st.error(f"Error: {e}")

if "suggestions" in st.session_state:
    st.subheader("Top 3 Suggested Topics")
    for idx, (title, reason) in enumerate(st.session_state["suggestions"], start=1):
        st.markdown(f"**{idx}) {title}**  \n_{reason}_")
    chosen = st.selectbox("Choose topic to generate script", [t for t, r in st.session_state["suggestions"]])
    if st.button("Run copyright & license safety check for chosen topic"):
        # perform LLM check + check licenses of top videos used as context
        with st.spinner("Running safety checks..."):
            # LLM check
            context_text = "\n".join([f"{v['title']} — {v['url']}" for v in st.session_state.get("videos", [])[:6]])
            status, explanation = llm_copyright_check(chosen, context_text)
            st.session_state["copyright_llm_status"] = (status, explanation)
            # YouTube license check on the fetched videos: if any video is creativeCommon -> safer
            youtube = googleapiclient.discovery.build(
                YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY
            )
            licenses = {}
            for v in st.session_state.get("videos", [])[:6]:
                lic = get_video_license(youtube, v["video_id"])
                licenses[v["video_id"]] = lic
            st.session_state["video_licenses"] = licenses
            st.success("Safety checks complete.")
    if "copyright_llm_status" in st.session_state:
        st.markdown("**LLM copyright check:**")
        st.write(st.session_state["copyright_llm_status"])
        st.markdown("**Top videos' YouTube license types (first 6):**")
        for vid, lic in st.session_state["video_licenses"].items():
            st.write(f"- {vid}: {lic}")
        # show recommendation
        llm_status, llm_reason = st.session_state["copyright_llm_status"]
        if llm_status == "RISK":
            st.warning(f"LLM flagged RISK: {llm_reason}. Consider picking another topic or ensure you use original content and royalty-free media.")
        else:
            st.success(f"LLM flagged SAFE: {llm_reason}")

    if st.button("Generate script for chosen topic"):
        with st.spinner("Generating script..."):
            script = generate_script_openai(chosen)
            st.session_state["script"] = script
            st.success("Script generated.")
            st.text_area("Generated Script", script, height=300)

if "script" in st.session_state:
    st.subheader("Create audio, video, and thumbnail")
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        if st.button("Generate Audio (gTTS)"):
            try:
                tmp_audio = Path(tempfile.gettempdir()) / f"{safe_file_name(chosen)}.mp3"
                text_to_speech_gtts(st.session_state["script"], tmp_audio)
                st.session_state["audio_path"] = str(tmp_audio)
                st.success(f"Audio created: {tmp_audio}")
            except Exception as e:
                st.error(f"Audio creation failed: {e}")
    with colB:
        num_images = st.number_input("Number of images for slideshow", min_value=2, max_value=12, value=6)
        if st.button("Assemble Video (slideshow + audio)"):
            if "audio_path" not in st.session_state:
                st.error("Generate audio first.")
            else:
                try:
                    with st.spinner("Fetching images and building video..."):
                        images = pexels_search_images(niche, per_page=int(num_images))
                        tmp_video = Path(tempfile.gettempdir()) / f"{safe_file_name(chosen)}.mp4"
                        build_video_from_script_and_images(st.session_state["script"], images, tmp_video, chosen)
                        st.session_state["video_path"] = str(tmp_video)
                        st.success(f"Video built: {tmp_video}")
                        st.video(str(tmp_video))
                except Exception as e:
                    st.error(f"Video assembly failed: {e}")
    with colC:
        if st.button("Generate Thumbnail (Pillow)"):
            try:
                tmp_thumb = Path(tempfile.gettempdir()) / f"{safe_file_name(chosen)}_thumb.jpg"
                generate_thumbnail_pillow(chosen, tmp_thumb, subtitle="Auto-generated")
                st.session_state["thumb_path"] = str(tmp_thumb)
                st.success(f"Thumbnail generated: {tmp_thumb}")
                st.image(str(tmp_thumb), width=480)
            except Exception as e:
                st.error(f"Thumbnail creation failed: {e}")

if "video_path" in st.session_state:
    st.subheader("Download assets")
    video_path = st.session_state["video_path"]
    thumb_path = st.session_state.get("thumb_path")
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    st.download_button("Download video (MP4)", data=video_bytes, file_name=Path(video_path).name, mime="video/mp4")
    if thumb_path:
        with open(thumb_path, "rb") as f:
            thumb_bytes = f.read()
        st.download_button("Download thumbnail (JPG)", data=thumb_bytes, file_name=Path(thumb_path).name, mime="image/jpeg")

st.caption("Notes: This app runs locally or on Render. For production Render deployments adapt storage (S3) and OAuth/token handling. MoviePy encoding is CPU-intensive—use an instance with adequate CPU for larger workloads.")
