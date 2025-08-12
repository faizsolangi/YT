import os
import tempfile
import time
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import streamlit as st
import requests
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy.video.VideoClip import TextClip, ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip, concatenate_videoclips
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import CompositeAudioClip, AudioClip as BaseAudioClip, AudioArrayClip
import numpy as np
import traceback
import openai
import googleapiclient.discovery
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from openai import OpenAI

# -------------------------
# Config & env variables
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")  # optional, for stock images

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_BASE_PATH = os.getenv("S3_BASE_PATH", "youtube-auto-studio/")
S3_PUBLIC_READ = os.getenv("S3_PUBLIC_READ", "true").lower() == "true"
S3_PRESIGN_EXPIRE_SECS = int(os.getenv("S3_PRESIGN_EXPIRE_SECS", "604800"))  # 7 days

if not OPENAI_API_KEY:
    st.error("Set OPENAI_API_KEY environment variable.")
    st.stop()
if not YOUTUBE_API_KEY:
    st.error("Set YOUTUBE_API_KEY environment variable.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
YOUTUBE_SEARCH_PUBLISHED_AFTER = os.getenv("YOUTUBE_SEARCH_PUBLISHED_AFTER", "2025-01-01T00:00:00Z")

# -------------------------
# Helper utilities
# -------------------------
def safe_file_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in " ._-" else "_" for c in s)[:120]


def fetch_trending_videos_from_youtube(query: str, max_results: int = 10, prefer_creative_commons: bool = False) -> List[dict]:
    youtube = googleapiclient.discovery.build(
        YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY
    )
    kwargs = {
        "q": query,
        "part": "snippet",
        "type": "video",
        "order": "viewCount",
        "maxResults": max_results,
        "publishedAfter": YOUTUBE_SEARCH_PUBLISHED_AFTER,
    }
    if prefer_creative_commons:
        kwargs["videoLicense"] = "creativeCommon"
    req = youtube.search().list(**kwargs)
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
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.35,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()


def parse_ranked_text(text: str) -> List[Tuple[str, str]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    results: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    for ln in lines:
        if ln[0].isdigit() and "Title:" in ln:
            title = ln.split("Title:", 1)[1].strip()
            current = {"title": title, "reason": ""}
            results.append(current)
        elif "Reason:" in ln and current is not None:
            current["reason"] = ln.split("Reason:", 1)[1].strip()
    return [(r["title"], r["reason"]) for r in results]


def llm_copyright_check(topic_title: str, context_text: str) -> Tuple[str, str]:
    """
    Ask LLM to classify risk: returns ("SAFE" or "RISK", explanation)
    """
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system = (
        "You are a cautious content-safety assistant. Classify whether creating a video on the given topic title risks "
        "copyright/trademark/rights-of-publicity or uses copyrighted characters/music/brands. Return exactly SAFE or "
        "RISK followed by a short reason."
    )
    user = (
        f"Topic title: {topic_title}\n\nContext (examples / related text):\n{context_text}\n\n"
        "Answer format:\nSAFE: short reason OR RISK: short reason"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=200,
    )
    out = resp.choices[0].message.content.strip()
    if out.upper().startswith("SAFE"):
        return "SAFE", out.split(":", 1)[1].strip() if ":" in out else ""
    elif out.upper().startswith("RISK"):
        return "RISK", out.split(":", 1)[1].strip() if ":" in out else ""
    else:
        return ("RISK", out) if "copyright" in out.lower() or "trademark" in out.lower() else ("SAFE", out)


def generate_script_openai(topic: str, target_seconds: Optional[int] = None) -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system = "You are a professional concise YouTube script writer. Produce a short script suitable for a 60-120s video."
    length_note = ""
    if target_seconds:
        approx_words = max(60, int(target_seconds * 2.4))
        length_note = (
            f"\nAim for about {target_seconds} seconds of narration (~{approx_words} words). "
            "Keep it tight and within that time."
        )
    user = (
        f"Write an engaging YouTube video script for the topic: \"{topic}\".\n"
        "- Hook (first 5-10s)\n- 3 short bullet points for main content\n- Quick summary\n- Call to action: like/subscribe\nKeep sentences short and energetic." + length_note
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.6,
        max_tokens=700,
    )
    return resp.choices[0].message.content.strip()


# -------------------------
# Image helpers (Pexels fallback)
# -------------------------
def pexels_search_images(query: str, per_page: int = 6) -> List[str]:
    if not PEXELS_API_KEY:
        return [
            "https://images.pexels.com/photos/3183186/pexels-photo-3183186.jpeg",
            "https://images.pexels.com/photos/1181675/pexels-photo-1181675.jpeg",
            "https://images.pexels.com/photos/3861969/pexels-photo-3861969.jpeg",
            "https://images.pexels.com/photos/414519/pexels-photo-414519.jpeg",
            "https://images.pexels.com/photos/842711/pexels-photo-842711.jpeg",
            "https://images.pexels.com/photos/1103539/pexels-photo-1103539.jpeg",
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
# Subtitles + TTS helpers (gTTS chunking)
# -------------------------
def split_text_into_chunks(text: str, max_chars: int = 180) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    # Split into sentences using simple regex
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            if len(sent) <= max_chars:
                current = sent
            else:
                # Hard wrap long sentences
                for i in range(0, len(sent), max_chars):
                    chunk = sent[i : i + max_chars]
                    if chunk:
                        chunks.append(chunk)
                current = ""
    if current:
        chunks.append(current)
    return chunks


def write_srt(srt_entries: List[Tuple[int, float, float, str]], out_path: Path) -> Path:
    # entries: (index starting at 1, start_sec, end_sec, text)
    def fmt_time(sec: float) -> str:
        millis = int(round((sec - int(sec)) * 1000))
        sec = int(sec)
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"

    lines: List[str] = []
    for idx, start_s, end_s, text in srt_entries:
        lines.append(str(idx))
        lines.append(f"{fmt_time(start_s)} --> {fmt_time(end_s)}")
        lines.append(text)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def text_to_speech_gtts(text: str, out_path: Path, lang: str = "en"):
    tts = gTTS(text, lang=lang)
    tts.save(str(out_path))
    return out_path


# -------------------------
# Text cleaning for TTS
# -------------------------
STAGE_KEYWORDS = r"music|sfx|sound|beat|applause|transition|intro|outro|fx|bgm|beatdrop|whoosh|ding|sting"

PAREN_STAGE_RE = re.compile(r"\((?:[^)]]*(?:" + STAGE_KEYWORDS + r")[^)]*)\)", flags=re.IGNORECASE)


def clean_script_for_tts(text: str) -> str:
    if not text:
        return ""
    # Remove markdown emphasis and headers/bullets
    text = re.sub(r"[*_#`]+", " ", text)
    text = re.sub(r"^\s*[-•]+\s+", "", text, flags=re.MULTILINE)
    # Remove bracketed stage directions like [INTRO], [SFX], [Hook]
    text = re.sub(r"\[[^\]]*\]", " ", text)
    # Remove parenthetical stage directions that include known keywords
    text = PAREN_STAGE_RE.sub(" ", text)
    # Also remove standalone lines that are likely cues (all caps short lines)
    text = re.sub(r"^\s*[A-Z ]{3,20}:?\s*$", " ", text, flags=re.MULTILINE)
    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pillow_fit_center_crop(img_path: str, width: int, height: int) -> np.ndarray:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        src_w, src_h = im.size
        target_ratio = width / height
        src_ratio = src_w / src_h
        if src_ratio >= target_ratio:
            new_h = height
            new_w = int(src_ratio * new_h)
        else:
            new_w = width
            new_h = int(new_w / src_ratio)
        im = im.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - width) // 2
        top = (new_h - height) // 2
        im = im.crop((left, top, left + width, top + height))
        return np.array(im)


def tts_chunks_and_srt(script_text: str, temp_dir: Path, lang: str = "en") -> Tuple[AudioArrayClip, Path, List[Tuple[int, float, float, str]]]:
    # Clean stage directions before chunking/tts
    script_text = clean_script_for_tts(script_text)
    chunks = split_text_into_chunks(script_text, max_chars=180)
    if not chunks:
        raise RuntimeError("No text to synthesize.")

    audio_arrays: List[np.ndarray] = []
    srt_entries: List[Tuple[int, float, float, str]] = []
    cursor = 0.0
    fps = 44100
    n_channels = 2
    for idx, chunk in enumerate(chunks, start=1):
        path = temp_dir / f"tts_{idx:03d}.mp3"
        text_to_speech_gtts(chunk, path, lang=lang)
        clip = AudioFileClip(str(path))
        try:
            arr = clip.to_soundarray(fps=fps)
            if arr.ndim == 1:
                arr = np.stack([arr, arr], axis=1)
            n_channels = arr.shape[1]
            audio_arrays.append(arr)
            start = cursor
            end = cursor + clip.duration
            srt_entries.append((idx, start, end, chunk))
            cursor = end
        finally:
            try:
                clip.close()
            except Exception:
                pass

    final_array = np.concatenate(audio_arrays, axis=0) if audio_arrays else np.zeros((1, n_channels), dtype=np.float32)
    final_audio = AudioArrayClip(final_array, fps=fps)
    srt_path = temp_dir / "subtitles.srt"
    write_srt(srt_entries, srt_path)
    return final_audio, srt_path, srt_entries


# -------------------------
# Title (Pillow fallback)
# -------------------------
def _pillow_title_clip(title_text: str, width: int, height: int, duration: float = 3.0,
                       bg_color=(20, 20, 20), font_color=(255, 255, 255)):
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 96)
    except Exception:
        try:
            font_title = ImageFont.truetype("DejaVuSans.ttf", 96)
        except Exception:
            font_title = ImageFont.load_default()

    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    max_text_width = width - 160
    words = (title_text or "").split()
    lines: List[str] = []
    current = ""
    for w in words:
        test = (current + " " + w).strip()
        if draw.textlength(test, font=font_title) <= max_text_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)

    line_height = getattr(font_title, "size", 60) + 10
    total_text_height = max(line_height, len(lines) * line_height)
    y = (height - total_text_height) // 2

    for line in lines[:5]:
        tw = draw.textlength(line, font=font_title)
        draw.text(((width - tw) / 2, y), line, font=font_title, fill=font_color)
        y += line_height

    return ImageClip(np.array(img), duration=duration)


# -------------------------
# Video assembly (MoviePy)
# -------------------------
def build_video_from_script_and_images(
    script_text: str,
    image_urls: List[str],
    out_video_path: Path,
    title_text: str,
    target_duration_s: Optional[float] = None,
    strict_enforce: bool = True,
) -> Tuple[Path, Path]:
    tmpdir = Path(tempfile.mkdtemp())

    # Create chunked audio + SRT (cleaned inside)
    final_audio, srt_path, srt_entries = tts_chunks_and_srt(script_text, tmpdir, lang="en")

    # Optionally enforce target duration on audio + SRT using array ops (avoid set_start/subclip)
    if target_duration_s and strict_enforce:
        fps = 44100
        arr = final_audio.to_soundarray(fps=fps)
        if arr.ndim == 1:
            arr = np.stack([arr, arr], axis=1)
        cur_samples = arr.shape[0]
        target_samples = int(float(target_duration_s) * fps)
        if cur_samples > target_samples:
            arr = arr[:target_samples]
            # Trim SRT
            target = float(target_duration_s)
            trimmed: List[Tuple[int, float, float, str]] = []
            for idx, start, end, text in srt_entries:
                if start >= target:
                    break
                trimmed.append((idx, start, min(end, target), text))
            srt_entries = trimmed
        elif cur_samples < target_samples:
            pad = target_samples - cur_samples
            n_channels = arr.shape[1] if arr.ndim == 2 else 2
            pad_arr = np.zeros((pad, n_channels), dtype=arr.dtype)
            arr = np.concatenate([arr, pad_arr], axis=0)
        final_audio = AudioArrayClip(arr, fps=fps)
        write_srt(srt_entries, srt_path)

    # Download images
    image_files: List[Path] = []
    for i, url in enumerate(image_urls):
        try:
            img_path = tmpdir / f"img_{i}.jpg"
            download_image(url, img_path)
            image_files.append(img_path)
        except Exception as e:
            print("Image download failed:", e)
    if not image_files:
        raise RuntimeError("No images to build video.")

    # Parameters for 1080p
    W, H = 1920, 1080

    # Title clip (3s) at 1080p
    try:
        title_clip = (
            TextClip(title_text, fontsize=80, color="white", size=(W, H), method="caption", font="DejaVu-Sans-Bold")
            .set_duration(3)
            .on_color(size=(W, H), color=(20, 20, 20))
        )
    except Exception:
        try:
            title_clip = (
                TextClip(title_text, fontsize=60, color="white", size=(W, H), method="caption")
                .set_duration(3)
                .on_color(size=(W, H), color=(20, 20, 20))
            )
        except Exception:
            title_clip = _pillow_title_clip(title_text, W, H, duration=3.0)

    # Build image clips; base durations on enforced audio duration if present
    n = len(image_files)
    base_duration = final_audio.duration
    per_clip = max(2.0, base_duration / max(1, n))
    clips: List[ImageClip] = []
    for idx, img in enumerate(image_files):
        frame = pillow_fit_center_crop(str(img), W, H)  # returns 1920x1080 np.ndarray
        clip = ImageClip(frame, duration=per_clip)
        clips.append(clip)
       




    slideshow = concatenate_videoclips(clips, method="compose")
    full = concatenate_videoclips([title_clip, slideshow], method="compose")

    # Ensure visual length >= audio; pad last frame if needed
    if full.duration < final_audio.duration:
        leftover = final_audio.duration - full.duration
        last_src = clips[-1]
        try:
            frame = last_src.get_frame(max(0.0, (last_src.duration or 0) - 1e-3))
        except Exception:
            frame = np.array(Image.open(str(image_files[-1])).convert("RGB"))
        last_pad = ImageClip(frame, duration=leftover)
        full = concatenate_videoclips([full, last_pad], method="compose")

    final = full.set_audio(final_audio).set_duration(final_audio.duration)

    # High-quality export settings
    final.write_videofile(
        str(out_video_path),
        fps=30,
        codec="libx264",
        audio_codec="aac",
        preset="slow",
        bitrate=None,
        ffmpeg_params=[
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-level",
            "4.2",
            "-movflags",
            "+faststart",
        ],
        threads=os.cpu_count() or 2,
    )

    return out_video_path, srt_path


# -------------------------
# Thumbnail generation (Pillow)
# -------------------------
def generate_thumbnail_pillow(title: str, out_path: Path, subtitle: str = ""):
    W, H = 1280, 720
    bg_color = (18, 18, 18)
    img = Image.new("RGB", (W, H), color=bg_color)
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 64)
        font_sub = ImageFont.truetype("DejaVuSans.ttf", 36)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    accent = Image.new("RGB", (W, 160), color=(255, 80, 80))
    img.paste(accent, (0, H - 160))

    max_width = W - 120
    words = title.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=font_title) <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    y = 140
    for line in lines[:3]:
        tw = draw.textlength(line, font=font_title)
        th = font_title.size + 6 if hasattr(font_title, "size") else 70
        draw.text(((W - tw) / 2, y), line, font=font_title, fill=(255, 255, 255))
        y += th

    if subtitle:
        sw = draw.textlength(subtitle, font=font_sub)
        draw.text(((W - sw) / 2, H - 120), subtitle, font=font_sub, fill=(255, 255, 255))

    img.save(out_path)
    return out_path


# -------------------------
# S3 storage helpers
# -------------------------
def get_s3_client():
    if not (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET_NAME):
        return None
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def s3_upload_file(local_path: Path, content_type: str = None) -> Tuple[bool, str]:
    s3 = get_s3_client()
    if not s3:
        return False, ""
    key = f"{S3_BASE_PATH}{safe_file_name(local_path.name)}"
    extra_args = {}
    if content_type:
        extra_args["ContentType"] = content_type
    if S3_PUBLIC_READ:
        extra_args["ACL"] = "public-read"
    try:
        s3.upload_file(str(local_path), S3_BUCKET_NAME, key, ExtraArgs=extra_args)
        if S3_PUBLIC_READ:
            url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"
            return True, url
        else:
            url = s3.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": S3_BUCKET_NAME, "Key": key},
                ExpiresIn=S3_PRESIGN_EXPIRE_SECS,
            )
            return True, url
    except (BotoCoreError, ClientError) as e:
        print("S3 upload failed:", e)
        return False, ""


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="YouTube Auto Studio (copyright + SRT + S3)", page_icon="🎬", layout="wide")
st.title("YouTube Auto Studio — Niche → Topic → Script → Video → Thumbnail")

with st.sidebar:
    st.header("Config")
    st.write("Ensure env vars:\n- OPENAI_API_KEY\n- YOUTUBE_API_KEY\n(optional) PEXELS_API_KEY\n(optional) AWS_ACCESS_KEY_ID/SECRET, S3_BUCKET_NAME")
    st.write("Note: This app builds videos and can upload to S3 for public links (good for Render).")
    safety_mode = st.checkbox("Safety mode (require LLM SAFE + creativeCommons)", value=True)
    prefer_cc = st.checkbox("Prefer Creative Commons in search", value=False)
    st.session_state["prefer_cc"] = prefer_cc
    debug_mode = st.checkbox("Debug mode (show full errors)", value=False)
    st.session_state["debug_mode"] = debug_mode
    confirm_royalty_free = st.checkbox("I confirm I will use only royalty-free/original media (e.g., Pexels, own assets)", value=False)
    st.session_state["confirm_royalty_free"] = confirm_royalty_free
    reuse_youtube_clips = st.checkbox("I will reuse clips from fetched YouTube videos", value=False)
    st.session_state["reuse_youtube_clips"] = reuse_youtube_clips

col1, col2 = st.columns([3, 1])
with col1:
    niche = st.text_input("Enter niche keyword (e.g., 'AI tools', 'solar energy'):", value="AI automation")
with col2:
    max_results = st.number_input("Search results", min_value=3, max_value=20, value=8, step=1)

if st.button("Fetch trending & suggest topics"):
    try:
        with st.spinner("Searching YouTube..."):
            vids = fetch_trending_videos_from_youtube(niche, max_results=int(max_results), prefer_creative_commons=bool(st.session_state.get("prefer_cc", False)))
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
    st.session_state["chosen_title"] = chosen

    # Length controls
    len_col1, len_col2 = st.columns([1, 1])
    with len_col1:
        length_mode = st.selectbox("Length preset", ["Short (~60s)", "Custom (seconds)"])
    with len_col2:
        if length_mode == "Custom (seconds)":
            target_seconds = int(st.number_input("Target length (seconds)", min_value=10, max_value=600, value=90, step=5))
        else:
            target_seconds = 60
    st.session_state["target_duration_s"] = target_seconds

    if st.button("Run copyright & license safety check for chosen topic"):
        with st.spinner("Running safety checks..."):
            context_text = "\n".join([f"{v['title']} — {v['url']}" for v in st.session_state.get("videos", [])])
            status, explanation = llm_copyright_check(chosen, context_text)
            st.session_state["copyright_llm_status"] = (status, explanation)
            youtube = googleapiclient.discovery.build(
                YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY
            )
            licenses: Dict[str, str] = {}
            for v in st.session_state.get("videos", []):
                lic = get_video_license(youtube, v["video_id"]) or "unknown"
                licenses[v["video_id"]] = lic
            st.session_state["video_licenses"] = licenses
            st.success("Safety checks complete.")
    if "copyright_llm_status" in st.session_state:
        st.markdown("**LLM copyright check:**")
        st.write(st.session_state["copyright_llm_status"])
        st.markdown("**Top videos' YouTube license types (first 6):**")
        for vid, lic in st.session_state.get("video_licenses", {}).items():
            st.write(f"- {vid}: {lic}")
        llm_status, llm_reason = st.session_state["copyright_llm_status"]
        if llm_status == "RISK":
            st.warning(f"LLM flagged RISK: {llm_reason}.")
        else:
            st.success(f"LLM flagged SAFE: {llm_reason}")

    def passes_safety_gate() -> bool:
        if not safety_mode:
            return True
        llm_status = st.session_state.get("copyright_llm_status", ("RISK", ""))[0]
        llm_safe = (llm_status == "SAFE")
        reuse = bool(st.session_state.get("reuse_youtube_clips", False))
        licenses = st.session_state.get("video_licenses", {})
        any_cc = any((lic or "").lower() == "creativecommon" for lic in licenses.values())
        if st.session_state.get("prefer_cc") and st.session_state.get("videos"):
            any_cc = True
        if st.session_state.get("confirm_royalty_free"):
            any_cc = True
        # If not reusing YouTube clips, CC is not required; only require LLM SAFE
        if not reuse:
            return llm_safe
        # If reusing YouTube clips, require both LLM SAFE and a CC source
        return llm_safe and any_cc

    if st.button("Generate script for chosen topic"):
        with st.spinner("Generating script..."):
            script = generate_script_openai(chosen, target_seconds=st.session_state.get("target_duration_s"))
            st.session_state["script"] = script
            st.success("Script generated.")
            st.text_area("Generated Script", script, height=300)
        if safety_mode and not passes_safety_gate():
            st.warning("Safety mode is ON: You can generate a script, but video assembly requires LLM SAFE and at least one creativeCommons result.")

if "script" in st.session_state:
    st.subheader("Create audio, video, subtitles, and thumbnail")
    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        if st.button("Generate Audio + SRT (chunked)"):
            if st.sidebar.checkbox("Re-run TTS", value=True, key="re_tts"):
                try:
                    tmpdir = Path(tempfile.gettempdir())
                    base = safe_file_name(st.session_state.get("script", "script"))
                    temp_audio = Path(tmpdir) / f"{base}_narration.mp3"
                    # Use cleaned text
                    cleaned = clean_script_for_tts(st.session_state["script"])
                    audio_clip, srt_path, _ = tts_chunks_and_srt(cleaned, Path(tempfile.mkdtemp()))
                    audio_clip.write_audiofile(str(temp_audio))
                    st.session_state["audio_path"] = str(temp_audio)
                    st.session_state["srt_path"] = str(srt_path)
                    st.success(f"Audio + SRT created: {temp_audio}, {srt_path}")
                except Exception as e:
                    st.error(f"Audio creation failed: {e}")
    with colB:
        num_images = st.number_input("Number of images for slideshow", min_value=2, max_value=12, value=6)
        strict_enforce = st.checkbox("Strictly enforce target duration", value=True)
        if st.button("Assemble Video (1080p + audio)"):
            if safety_mode and not passes_safety_gate():
                # Auto-run safety checks once if missing, then retry gate
                if "copyright_llm_status" not in st.session_state or "video_licenses" not in st.session_state:
                    try:
                        with st.spinner("Running safety checks..."):
                            context_text = "\n".join([f"{v['title']} — {v['url']}" for v in st.session_state.get("videos", [])])
                            status, explanation = llm_copyright_check(st.session_state.get("chosen_title") or chosen, context_text)
                            st.session_state["copyright_llm_status"] = (status, explanation)
                            youtube = googleapiclient.discovery.build(
                                YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY
                            )
                            licenses: Dict[str, str] = {}
                            for v in st.session_state.get("videos", []):
                                lic = get_video_license(youtube, v["video_id"]) or "unknown"
                                licenses[v["video_id"]] = lic
                            st.session_state["video_licenses"] = licenses
                    except Exception as e:
                        st.warning(f"Safety re-check error: {e}")
                if safety_mode and not passes_safety_gate():
                    reuse = bool(st.session_state.get("reuse_youtube_clips", False))
                    llm_status = st.session_state.get("copyright_llm_status", ("RISK", ""))[0]
                    if not reuse:
                        if llm_status != "SAFE":
                            st.error("Safety mode is ON: LLM must be SAFE before assembling. Run safety check.")
                        else:
                            st.error("Safety mode is ON: Policy blocked. Disable Safety mode to proceed.")
                    else:
                        st.error("Safety mode is ON: To reuse YouTube clips, requires LLM SAFE and at least one creativeCommons source. Toggle 'Prefer Creative Commons in search', re-fetch, or uncheck 'reuse YouTube clips'.")
                    st.stop()
            try:
                with st.spinner("Fetching images and building video..."):
                    images = pexels_search_images(niche, per_page=int(num_images))
                    tmp_video = Path(tempfile.gettempdir()) / f"{safe_file_name(st.session_state.get('script','video'))}.mp4"
                    video_path, srt_path = build_video_from_script_and_images(
                        st.session_state["script"],
                        images,
                        tmp_video,
                        title_text=st.session_state.get("chosen_title", "") or safe_file_name(niche),
                        target_duration_s=st.session_state.get("target_duration_s"),
                        strict_enforce=bool(strict_enforce),
                    )
                    st.session_state["video_path"] = str(video_path)
                    st.session_state["srt_path"] = str(srt_path)
                    st.success(f"Video built: {tmp_video}")
                    st.video(str(tmp_video))
            except Exception as e:
                st.error(f"Video assembly failed: {e}")
                if st.session_state.get("debug_mode"):
                    st.exception(e)
                    st.code(traceback.format_exc())
    with colC:
        if st.button("Generate Thumbnail (Pillow)"):
            try:
                tmp_thumb = Path(tempfile.gettempdir()) / f"{safe_file_name(niche)}_thumb.jpg"
                generate_thumbnail_pillow(safe_file_name(niche)[:80], tmp_thumb, subtitle="Auto-generated")
                st.session_state["thumb_path"] = str(tmp_thumb)
                st.success(f"Thumbnail generated: {tmp_thumb}")
                st.image(str(tmp_thumb), width=480)
            except Exception as e:
                st.error(f"Thumbnail creation failed: {e}")

if "video_path" in st.session_state or "audio_path" in st.session_state or "thumb_path" in st.session_state or "srt_path" in st.session_state:
    st.subheader("Download or S3 links")
    video_path = st.session_state.get("video_path")
    audio_path = st.session_state.get("audio_path")
    srt_path = st.session_state.get("srt_path")
    thumb_path = st.session_state.get("thumb_path")

    # Local download buttons
    if video_path and Path(video_path).exists():
        with open(video_path, "rb") as f:
            st.download_button("Download video (MP4)", data=f.read(), file_name=Path(video_path).name, mime="video/mp4")
    if audio_path and Path(audio_path).exists():
        with open(audio_path, "rb") as f:
            st.download_button("Download audio (MP3)", data=f.read(), file_name=Path(audio_path).name, mime="audio/mpeg")
    if srt_path and Path(srt_path).exists():
        with open(srt_path, "rb") as f:
            st.download_button("Download subtitles (SRT)", data=f.read(), file_name=Path(srt_path).name, mime="application/x-subrip")
    if thumb_path and Path(thumb_path).exists():
        with open(thumb_path, "rb") as f:
            st.download_button("Download thumbnail (JPG)", data=f.read(), file_name=Path(thumb_path).name, mime="image/jpeg")

    # S3 upload and public links
    if S3_BUCKET_NAME and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        st.write("S3 uploads:")
        up_results = []
        if video_path and Path(video_path).exists():
            ok, url = s3_upload_file(Path(video_path), content_type="video/mp4")
            if ok:
                st.write(f"- Video: {url}")
        if audio_path and Path(audio_path).exists():
            ok, url = s3_upload_file(Path(audio_path), content_type="audio/mpeg")
            if ok:
                st.write(f"- Audio: {url}")
        if srt_path and Path(srt_path).exists():
            ok, url = s3_upload_file(Path(srt_path), content_type="application/x-subrip")
            if ok:
                st.write(f"- Subtitles: {url}")
        if thumb_path and Path(thumb_path).exists():
            ok, url = s3_upload_file(Path(thumb_path), content_type="image/jpeg")
            if ok:
                st.write(f"- Thumbnail: {url}")

st.caption(
    "Notes: This app can run locally or on Render. Large files are uploaded to S3 if configured. MoviePy encoding is CPU-intensive—use an instance with adequate CPU."
)