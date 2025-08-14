import os
import tempfile
import re
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import streamlit as st
import requests
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import traceback
import gc
import googleapiclient.discovery
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from openai import OpenAI
from pytrends.request import TrendReq
from requests.exceptions import RequestException

# MoviePy submodule imports (avoid moviepy.editor for compatibility)
from moviepy.video.VideoClip import TextClip, ImageClip, VideoClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

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

def concatenate_videoclips_simple(clips, size=None):
    if not clips:
        raise ValueError("No clips to concatenate")
    durations = [float(getattr(c, "duration", 0) or 0) for c in clips]
    total_duration = float(sum(durations))

    def make_frame(t):
        if t <= 0:
            return clips[0].get_frame(0)
        if t >= total_duration:
            # guard for exact end
            return clips[-1].get_frame(max(0.0, durations[-1] - 1e-6))
        acc = 0.0
        for i, d in enumerate(durations):
            if t < acc + d:
                return clips[i].get_frame(t - acc)
            acc += d
        return clips[-1].get_frame(max(0.0, durations[-1] - 1e-6))

    composite = VideoClip(make_frame=make_frame, duration=total_duration)
    # Let writer infer size from first frame; optionally set size if provided
    if size and len(size) == 2 and all(size):
        try:
            composite.w, composite.h = size[0], size[1]
        except Exception:
            pass
    return composite


def safe_file_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in " ._-" else "_" for c in s)[:120]


def iso_timestamp_days_ago(days: int) -> str:
    from datetime import datetime, timedelta, timezone
    dt = datetime.now(timezone.utc) - timedelta(days=max(0, days))
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_trending_videos_from_youtube(query: str, max_results: int = 10, prefer_creative_commons: bool = False, published_after: Optional[str] = None) -> List[dict]:
    youtube = googleapiclient.discovery.build(
        YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY
    )
    kwargs = {
        "q": query,
        "part": "snippet",
        "type": "video",
        "order": "viewCount",
        "maxResults": max_results,
        "publishedAfter": published_after or YOUTUBE_SEARCH_PUBLISHED_AFTER,
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


def call_openai_rank_topics_combined(niche: str, videos: List[dict], trends: Dict[str, Any], days: int = 7) -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    yt_text = "\n".join([f"- {v['title']} — {v['url']} — {v['channel']}" for v in videos[:12]])
    top_queries = trends.get("related_queries_top") or []
    rising_queries = trends.get("related_queries_rising") or []
    suggestions = trends.get("suggestions") or []

    def topn_list(df_like, n=10):
        try:
            # pytrends returns pandas DataFrame-like objects
            rows = df_like.head(n).to_dict(orient="records")
            return [str(r.get("query") or r) for r in rows]
        except Exception:
            return []

    q_top = topn_list(top_queries, 8) if hasattr(top_queries, "head") else []
    q_rising = topn_list(rising_queries, 8) if hasattr(rising_queries, "head") else []
    sug_titles = [s.get("title") for s in suggestions if isinstance(s, dict)]

    trends_snippet = "\n".join([f"- {q}" for q in (q_top + q_rising + (sug_titles or []))[:12]])

    system = (
        "You are a concise YouTube trend strategist. Combine viral YouTube content from the past week with Google Trends cues to craft topics likely to perform in the next 2 weeks."
    )
    user = (
        f"Niche: {niche}\n\nYouTube (last {days} days, top by views):\n{yt_text or '- none'}\n\nGoogle Trends related queries/suggestions:\n{trends_snippet or '- none'}\n\nReturn exactly three suggested topic headlines with one short reason each.\nFormat:\n1) Title: <title>\n   Reason: <one short reason>\n2) Title: ...\n3) Title: ..."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.4,
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
                for i in range(0, len(sent), max_chars):
                    chunk = sent[i : i + max_chars]
                    if chunk:
                        chunks.append(chunk)
                current = ""
    if current:
        chunks.append(current)
    return chunks


def write_srt(srt_entries: List[Tuple[int, float, float, str]], out_path: Path) -> Path:
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

PAREN_STAGE_RE = re.compile(r"\((?:[^)]*(?:" + STAGE_KEYWORDS + r")[^)]*)\)", flags=re.IGNORECASE)


def clean_script_for_tts(text: str) -> str:
    if not text:
        return ""
    # Remove markdown/formatting
    text = re.sub(r"[*_#`]+", " ", text)
    text = re.sub(r"^\s*[-•]+\s+", "", text, flags=re.MULTILINE)
    # Remove bracketed or parenthetical stage directions
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = PAREN_STAGE_RE.sub(" ", text)
    # Remove speaker labels like "Host:", "Narrator:", "VO:", etc. at line starts
    text = re.sub(r"^\s*[A-Za-z][A-Za-z ]{0,20}:\s+", " ", text, flags=re.MULTILINE)
    # Remove common sound cue phrases inline
    text = re.sub(r"\b(jingle\s*bell\s*sound|applause|whoosh|sting|ding|sfx|bgm|music\s*(starts|plays)?|sound\s*effect[s]?)\b[:,-]*\s*", " ", text, flags=re.IGNORECASE)
    # Remove standalone short all-caps cue lines
    text = re.sub(r"^\s*[A-Z ]{3,20}:?\s*$", " ", text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tts_chunks_and_srt(script_text: str, temp_dir: Path, lang: str = "en") -> Tuple[AudioArrayClip, Path, List[Tuple[int, float, float, str]]]:
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
# Pillow helpers for video frames
# -------------------------

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


def _pillow_title_frame(title_text: str, width: int, height: int,
                        bg_color=(20, 20, 20), font_color=(255, 255, 255)) -> np.ndarray:
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

    return np.array(img)


def draw_subtitle_on_frame(base_frame: np.ndarray, text: str, width: int, height: int) -> np.ndarray:
    if not text:
        return base_frame
    img = Image.fromarray(base_frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 36)
    except Exception:
        font = ImageFont.load_default()

    padding_x, padding_y = 24, 16
    max_text_width = width - 2 * padding_x

    # Wrap text by measuring width
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=font) <= max_text_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    line_spacing = 8
    line_height = (getattr(font, "size", 36)) + line_spacing
    total_text_h = len(lines) * line_height - line_spacing

    # Box dimensions and position (bottom area)
    box_margin_bottom = 40
    box_w = max(draw.textlength(line, font=font) for line in lines) + 2 * padding_x
    box_h = total_text_h + 2 * padding_y
    box_x = (width - box_w) // 2
    box_y = height - box_h - box_margin_bottom

    # Semi-transparent black box
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rounded_rectangle([box_x, box_y, box_x + box_w, box_y + box_h], radius=12, fill=(0, 0, 0, 170))

    # Draw text with white fill and black stroke
    ty = box_y + padding_y
    for line in lines:
        tw = draw.textlength(line, font=font)
        tx = int((width - tw) / 2)
        odraw.text((tx, ty), line, font=font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        ty += line_height

    composed = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return np.array(composed)


def build_timeline_frames(
    title_frame: np.ndarray,
    slide_frames: List[np.ndarray],
    title_duration: float,
    per_clip: float,
    total_duration: float,
    srt_entries: List[Tuple[int, float, float, str]],
    width: int,
    height: int,
) -> Tuple[List[np.ndarray], List[float]]:
    # Build slide boundary times
    times = [0.0]
    t = 0.0
    t += float(title_duration)
    times.append(t)
    for _ in slide_frames:
        t += float(per_clip)
        times.append(min(t, total_duration))
        if t >= total_duration:
            break
    # Ensure final boundary equals total_duration
    if times[-1] < total_duration:
        times[-1] = total_duration

    # Collect SRT boundaries
    for _, s, e, _ in srt_entries:
        if 0.0 < s < total_duration:
            times.append(s)
        if 0.0 < e < total_duration:
            times.append(e)

    # Unique sorted boundaries
    boundaries = sorted(set(round(x, 3) for x in times))
    if boundaries[0] != 0.0:
        boundaries = [0.0] + boundaries
    if boundaries[-1] != round(total_duration, 3):
        boundaries.append(round(total_duration, 3))

    # Helper to select slide frame at time t
    def frame_at(tcur: float) -> np.ndarray:
        if tcur < title_duration:
            return title_frame
        idx = int((tcur - title_duration) // per_clip)
        if idx < 0:
            idx = 0
        if idx >= len(slide_frames):
            idx = len(slide_frames) - 1
        return slide_frames[idx]

    # Helper to active subtitle at time t
    def subtitle_at(tcur: float) -> str:
        for _, s, e, text in srt_entries:
            if s <= tcur < e:
                return text
        return ""

    frames: List[np.ndarray] = []
    durations: List[float] = []

    for i in range(len(boundaries) - 1):
        start = float(boundaries[i])
        end = float(boundaries[i + 1])
        if end <= start:
            continue
        base = frame_at(start)
        sub_text = subtitle_at(start)
        out_frame = draw_subtitle_on_frame(base, sub_text, width, height) if sub_text else base
        frames.append(out_frame)
        durations.append(end - start)

    # Normalize to exact total_duration if minor rounding issues
    dur_sum = sum(durations)
    if abs(dur_sum - total_duration) > 1e-3:
        durations[-1] += (total_duration - dur_sum)

    return frames, durations


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
    motion_enabled: bool = False,
) -> Tuple[Path, Path]:
    tmpdir = Path(tempfile.mkdtemp())

    # Create chunked audio + SRT (cleaned inside)
    final_audio, srt_path, srt_entries = tts_chunks_and_srt(script_text, tmpdir, lang="en")

    # Optionally enforce target duration on audio + SRT using array ops
    if target_duration_s and strict_enforce:
        fps = 44100
        arr = final_audio.to_soundarray(fps=fps)
        if arr.ndim == 1:
            arr = np.stack([arr, arr], axis=1)
        cur_samples = arr.shape[0]
        target_samples = int(float(target_duration_s) * fps)
        if cur_samples > target_samples:
            arr = arr[:target_samples]
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

    # Parameters for 720p to reduce memory/CPU
    W, H = 1280, 720

    # No title frame (removed per request)

    # Build image frames; base durations on enforced audio duration if present
    n = len(image_files)
    base_duration = final_audio.duration
    per_clip = max(2.0, base_duration / max(1, n))

    frames: List[np.ndarray] = []
    durations: List[float] = []  # kept for reference only

    # Choose a low fps to keep frame list small while allowing second-level granularity
    seq_fps = 24  # target fps for durations-based sequence (ImageSequenceClip will handle durations)

    # Build slideshow with optional motion
    slide_frames = [pillow_fit_center_crop(str(img), W, H) for img in image_files]
    frames: List[np.ndarray] = []
    durations: List[float] = []
    if motion_enabled:
        for sf in slide_frames:
            try:
                mf, md = ken_burns_frames(sf, W, H, duration=float(per_clip), fps=12)
            except NameError:
                mf, md = [sf], [float(per_clip)]
            frames.extend(mf)
            durations.extend(md)
    else:
        frames = slide_frames
        durations = [per_clip] * len(slide_frames)

    # Adjust last duration to match audio length exactly
    total = float(sum(durations))
    audio_total = float(final_audio.duration)
    if total < audio_total and durations:
        durations[-1] += (audio_total - total)
    elif total > audio_total and durations and durations[-1] > (total - audio_total + 0.1):
        durations[-1] -= (total - audio_total)

    video = ImageSequenceClip(frames, durations=durations)

    try:
        final = video.set_audio(final_audio)
    except Exception:
        video.audio = final_audio
        final = video

    final.write_videofile(
        str(out_video_path),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        preset="veryfast",
        bitrate=None,
        ffmpeg_params=[
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ],
        threads=1,
    )

    return out_video_path, srt_path


# -------------------------
# Thumbnail generation (Pillow)
# -------------------------

def generate_thumbnail_pillow(title: str, out_path: Path, subtitle: str = ""):
    W, H = 1280, 720
    bg_color = (16, 16, 16)
    img = Image.new("RGB", (W, H), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Try high-impact fonts and larger sizes
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 96)
    except Exception:
        try:
            font_title = ImageFont.truetype("DejaVuSans.ttf", 96)
        except Exception:
            font_title = ImageFont.load_default()
    try:
        font_sub = ImageFont.truetype("DejaVuSans.ttf", 44)
    except Exception:
        font_sub = ImageFont.load_default()

    # Keep a clean black background for contrast, add subtle top glow
    for y in range(80):
        ratio = y / 80.0
        val = int(16 + ratio * 32)
        draw.line([(0, y), (W, y)], fill=(val, val, val))

    # Accent bar (optional subtle)
    accent = Image.new("RGB", (W, 8), color=(255, 64, 64))
    img.paste(accent, (0, H - 140))

    # Title wrapping with tighter bounds for bigger font
    max_width = W - 160
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

    # Shadowed text for readability
    y = 110
    for line in lines[:3]:
        tw = draw.textlength(line, font=font_title)
        x = int((W - tw) / 2)
        # shadow
        draw.text((x + 3, y + 3), line, font=font_title, fill=(0, 0, 0))
        draw.text((x, y), line, font=font_title, fill=(255, 255, 255))
        y += (getattr(font_title, "size", 96) + 8)

    if subtitle:
        sw = draw.textlength(subtitle, font=font_sub)
        sx = int((W - sw) / 2)
        sy = H - 120
        draw.text((sx + 2, sy + 2), subtitle, font=font_sub, fill=(0, 0, 0))
        draw.text((sx, sy), subtitle, font=font_sub, fill=(255, 255, 255))

    # Save with high quality
    img.save(out_path, quality=95, subsampling=0, optimize=True)
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
# Google Trends + SEO metadata
# -------------------------

def fetch_google_trends(niche: str, geo: str = "US", timeframe: str = "today 3-m", lang: str = "en-US", prefer_youtube: bool = False, aggressive: bool = True) -> Dict[str, Any]:
    proxies_env = os.getenv("PYTRENDS_PROXIES", "").strip()
    proxies = None
    if proxies_env:
        try:
            proxies = json.loads(proxies_env)
        except Exception:
            proxies = {"https": proxies_env, "http": proxies_env}

    def make_pytrends():
        return TrendReq(hl=lang, tz=360, proxies=proxies)

    kw = niche.strip() or "trending"

    # Build fallback lists
    timeframe_list = [timeframe]
    if aggressive:
        for tf in ["today 12-m", "today 5-y", "all"]:
            if tf not in timeframe_list:
                timeframe_list.append(tf)
    gprops = ["youtube", ""] if prefer_youtube else ["", "youtube"]

    backoff = 1.0

    def try_call(fn, *args, **kwargs):
        nonlocal backoff
        try:
            res = fn(*args, **kwargs)
            time.sleep(0.6)
            return res
        except Exception:
            time.sleep(backoff + np.random.rand() * 0.4)
            backoff = min(backoff * 1.7 + 0.2, 20.0)
            return None

    def nonzero_series(iot_df) -> bool:
        try:
            if iot_df is None or getattr(iot_df, "empty", True):
                return False
            if kw not in iot_df.columns:
                return False
            col = iot_df[kw]
            return bool(np.any((np.array(col.values, dtype=float) > 0)))
        except Exception:
            return False

    last_err = None
    chosen = None
    pytrends = None

    for tf in timeframe_list:
        for gp in gprops:
            for attempt in range(4):
                try:
                    pytrends = make_pytrends()
                    pytrends.build_payload([kw], cat=0, timeframe=tf, geo=geo, gprop=gp)
                    iot = try_call(pytrends.interest_over_time)
                    if nonzero_series(iot):
                        chosen = (tf, gp)
                        break
                    # Try best region if zero
                    ibr = try_call(pytrends.interest_by_region, resolution='COUNTRY', inc_low_vol=True)
                    if ibr is not None and not getattr(ibr, 'empty', True):
                        ibr_sorted = ibr.sort_values(by=kw, ascending=False)
                        if not ibr_sorted.empty and float(ibr_sorted.iloc[0][kw]) > 0:
                            top_region = getattr(ibr_sorted.iloc[0], ibr_sorted.index.name or 'geoName') if hasattr(ibr_sorted, 'index') else None
                            if top_region and isinstance(top_region, str) and len(top_region) <= 2:
                                # likely ISO-2 code
                                pytrends = make_pytrends()
                                pytrends.build_payload([kw], cat=0, timeframe=tf, geo=top_region, gprop=gp)
                                iot2 = try_call(pytrends.interest_over_time)
                                if nonzero_series(iot2):
                                    geo = top_region
                                    iot = iot2
                                    chosen = (tf, gp)
                                    break
                    last_err = None
                except Exception as e:
                    last_err = e
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 16.0)
            if chosen:
                break
        if chosen:
            break

    if not chosen:
        # Fallback minimal structure when zero throughout
        return {
            "keyword": kw,
            "geo": geo,
            "timeframe": timeframe,
            "gprop": gprops[0] if gprops else "",
            "related_queries_top": [],
            "related_queries_rising": [],
            "related_topics": [],
            "interest_over_time": [],
            "suggestions": [],
            "error": f"no_nonzero_series (last_err={last_err})",
        }

    # With chosen settings, collect details
    data: Dict[str, Any] = {"keyword": kw, "geo": geo, "timeframe": chosen[0], "gprop": chosen[1]}

    rq = try_call(pytrends.related_queries)
    if isinstance(rq, dict):
        data["related_queries_top"] = (rq.get(kw, {}) or {}).get("top")
        data["related_queries_rising"] = (rq.get(kw, {}) or {}).get("rising")
    else:
        data["related_queries_top"] = []
        data["related_queries_rising"] = []

    topics = try_call(pytrends.related_topics)
    data["related_topics"] = topics.get(kw) if isinstance(topics, dict) else []

    iot = try_call(pytrends.interest_over_time)
    if iot is not None and hasattr(iot, "empty") and not iot.empty and kw in iot.columns:
        series = iot[kw].reset_index().rename(columns={kw: "score"})
        data["interest_over_time"] = series.to_dict(orient="records")
    else:
        data["interest_over_time"] = []

    sug = try_call(pytrends.suggestions, keyword=kw)
    data["suggestions"] = sug if isinstance(sug, list) else []

    return data


def analyze_trend_strength(interest_points: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not interest_points:
        return {"direction": "unknown", "slope": 0.0}
    y = np.array([float(p.get("score", 0.0)) for p in interest_points], dtype=float)
    x = np.arange(len(y), dtype=float)
    if len(y) < 2 or np.all(y == y[0]):
        return {"direction": "flat", "slope": 0.0}
    slope = float(np.polyfit(x, y, 1)[0])
    direction = "rising" if slope > 0 else ("falling" if slope < 0 else "flat")
    return {"direction": direction, "slope": slope}


def generate_metadata_openai(niche: str, trends: Dict[str, Any]) -> Dict[str, Any]:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    trend_brief = {
        "keyword": trends.get("keyword"),
        "geo": trends.get("geo"),
        "timeframe": trends.get("timeframe"),
        "related_queries_top": (trends.get("related_queries_top") or [])[:10],
        "related_queries_rising": (trends.get("related_queries_rising") or [])[:10],
        "suggestions": (trends.get("suggestions") or [])[:10],
        "interest_summary": analyze_trend_strength(trends.get("interest_over_time") or []),
    }

    system = (
        "You are an SEO optimizer for YouTube. Using the provided trends data, craft: "
        "1 best video title (<= 70 chars), 3 alternative titles, 10-15 comma-separated SEO tags, "
        "and a compelling description (<= 200 words). Predict if the topic is rising/flat/falling and explain briefly. "
        "Return only strict JSON with keys: best_title, alt_titles (list), tags (list), description, predicted_trend, reasoning."
    )
    user = json.dumps({"niche": niche, "trends": trend_brief}, ensure_ascii=False)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.4,
        max_tokens=700,
    )
    content = resp.choices[0].message.content.strip()
    try:
        data = json.loads(content)
    except Exception:
        # Fallback minimal structure
        data = {
            "best_title": f"{niche}: What You Need to Know Right Now",
            "alt_titles": [f"{niche} Trends Explained", f"{niche} in 60 Seconds", f"Top {niche} Tips"],
            "tags": [niche, f"{niche} tips", f"{niche} trends"],
            "description": content[:900],
            "predicted_trend": analyze_trend_strength(trends.get("interest_over_time") or []).get("direction", "unknown"),
            "reasoning": "Generated without strict JSON parse; please review.",
        }
    return data


def compute_trend_score(interest_points: List[Dict[str, Any]], method: str = "max", window: int = 7) -> float:
    if not interest_points:
        return 0.0
    scores = [float(p.get("score", 0.0)) for p in interest_points]
    if not scores:
        return 0.0
    if method == "recent_avg" and window and window > 1 and len(scores) >= window:
        return float(np.mean(scores[-window:]))
    return float(np.max(scores))


def pytrends_suggestions_list(keyword: str, lang: str = "en-US") -> List[str]:
    try:
        tr = TrendReq(hl=lang, tz=360)
        sug = tr.suggestions(keyword=keyword) or []
        titles = []
        for item in sug:
            title = item.get("title") or item.get("keyword") or ""
            if title and title.lower() != keyword.lower():
                titles.append(title)
        return titles
    except Exception:
        return []


def generate_keyword_variants(niche: str) -> List[str]:
    niche = (niche or "").strip()
    year = str((__import__("datetime").datetime.utcnow().year))
    seeds = [niche]
    modifiers = ["trends", "news", "guide", "ideas", "tips", "best", "tools", year]
    for m in modifiers:
        if m and niche:
            seeds.append(f"{niche} {m}")
    return list(dict.fromkeys([s.strip() for s in seeds if s.strip()]))


def find_trending_keyword(
    niche: str,
    min_score: float = 50.0,
    geo: str = "US",
    timeframe: str = "today 3-m",
    lang: str = "en-US",
    max_attempts: int = 20,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], float]:
    candidates: List[str] = []
    # Suggestions from pytrends
    suggestions = pytrends_suggestions_list(niche, lang=lang)
    candidates.extend(suggestions)
    # Variants
    candidates.extend(generate_keyword_variants(niche))
    # Unique preserve order
    seen = set()
    uniq: List[str] = []
    for c in candidates:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    if not uniq:
        uniq = [niche]

    attempts = 0
    best_kw = None
    best_data = None
    best_score = -1.0

    for kw in uniq:
        if attempts >= max_attempts:
            break
        attempts += 1
        data = fetch_google_trends(kw, geo=geo, timeframe=timeframe, lang=lang)
        score = compute_trend_score(data.get("interest_over_time") or [], method="max")
        if score > best_score:
            best_kw, best_data, best_score = kw, data, score
        if score >= float(min_score):
            return kw, data, score
    return best_kw, best_data, float(best_score)


# Move AI image helper functions here so they are defined before usage in the UI

def chunk_script_for_images(text: str, max_chunks: int = 8) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []
    chunks: List[str] = []
    cur = ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= 180:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
        if len(chunks) >= max_chunks:
            break
    if cur and len(chunks) < max_chunks:
        chunks.append(cur)
    return chunks


def prompts_from_script_chunks(chunks: List[str], style: str = "cartoon, vibrant, friendly, flat shading") -> List[str]:
    prompts: List[str] = []
    for i, ch in enumerate(chunks, 1):
        prompts.append(
            f"Illustration, {style}. Scene {i}: {ch}. Clear composition, single focal subject, no text, 16:9."
        )
    return prompts


def openai_generate_image(prompt: str, out_path: Path, size: str = "1280x720") -> Path:
    model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
    # Stream small size first for memory safety; upscale with Pillow if needed
    small_size = os.getenv("OPENAI_IMAGE_SMALL", "512x288")
    try:
        resp = client.images.generate(model=model, prompt=prompt, size=small_size)
        b64 = resp.data[0].b64_json
    except Exception:
        # fallback to default size if small not supported
        resp = client.images.generate(model=model, prompt=prompt, size=size)
        b64 = resp.data[0].b64_json
    import base64
    data = base64.b64decode(b64)
    with open(out_path, "wb") as f:
        f.write(data)
    # If generated small, upscale to target with Pillow (low memory)
    try:
        with Image.open(out_path) as im:
            im = im.convert("RGB").resize(tuple(map(int, size.split("x"))), Image.LANCZOS)
            im.save(out_path, quality=92, optimize=True)
    except Exception:
        pass
    return out_path


def generate_cartoon_images_from_script(script_text: str, temp_dir: Path, max_images: int = 8) -> List[Path]:
    cleaned = clean_script_for_tts(script_text)
    chunks = chunk_script_for_images(cleaned, max_chunks=max_images)
    prompts = prompts_from_script_chunks(chunks)
    paths: List[Path] = []
    for i, p in enumerate(prompts, 1):
        path = temp_dir / f"ai_img_{i:02d}.jpg"
        try:
            openai_generate_image(p, path, size="1280x720")
            paths.append(path)
        except Exception as e:
            print("AI image generation failed:", e)
        finally:
            if i % 2 == 0:
                gc.collect()
    return paths


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="YouTube Auto Studio (Trends + SEO + SRT + S3)", page_icon="🎬", layout="wide")
st.title("YouTube Auto Studio — Trends → Topic → Script → Video → SEO → Thumbnail")

with st.sidebar:
    st.header("Config")
    st.write("Ensure env vars:\n- OPENAI_API_KEY\n- YOUTUBE_API_KEY\n(optional) PEXELS_API_KEY\n(optional) AWS_ACCESS_KEY_ID/SECRET, S3_BUCKET_NAME")
    st.write("Note: App builds videos and can upload to S3 for public links.")
    safety_mode = st.checkbox("Safety mode (require LLM SAFE + creativeCommons)", value=True)
    prefer_cc = st.checkbox("Prefer Creative Commons in search", value=False)
    st.session_state["prefer_cc"] = prefer_cc
    debug_mode = st.checkbox("Debug mode (show full errors)", value=False)
    st.session_state["debug_mode"] = debug_mode
    confirm_royalty_free = st.checkbox("I confirm I will use only royalty-free/original media", value=False)
    st.session_state["confirm_royalty_free"] = confirm_royalty_free
    reuse_youtube_clips = st.checkbox("I will reuse clips from fetched YouTube videos", value=False)
    st.session_state["reuse_youtube_clips"] = reuse_youtube_clips

    st.markdown("---")
    st.subheader("Google Trends")
    geo = st.selectbox("Region", ["US", "GB", "IN", "CA", "AU", "DE", "FR", "BR", "ZA", "JP", "KR", "RU", "MX", "IT", "ES", "NL", "SE", "NO", "PL", "TR", "ID", "PH", "VN", "SG", "AE", "SA"], index=0)
    timeframe = st.selectbox("Timeframe", ["now 7-d", "today 1-m", "today 3-m", "today 12-m", "today 5-y", "all"], index=2)
    min_trend_score = st.slider("Minimum trend score to accept", min_value=0, max_value=100, value=50, step=5)
    virality_days = st.slider("Virality window (days)", min_value=1, max_value=30, value=7, step=1)
    st.session_state["trends_geo"] = geo
    st.session_state["trends_timeframe"] = timeframe
    st.session_state["min_trend_score"] = min_trend_score
    st.session_state["virality_days"] = virality_days
    use_ai_imgs = st.checkbox("Use AI-generated images for video", value=False, key="use_ai_imgs")

col1, col2 = st.columns([3, 1])
with col1:
    niche = st.text_input("Enter niche keyword (e.g., 'AI tools', 'solar energy'):", value="AI automation")
with col2:
    max_results = st.number_input("Search results", min_value=3, max_value=20, value=8, step=1)

if st.button("Fetch trending & suggest topics"):
    try:
        with st.spinner("Searching YouTube..."):
            past_days = int(st.session_state.get("virality_days", 7))
            vids = fetch_trending_videos_from_youtube(
                niche,
                max_results=int(max_results),
                prefer_creative_commons=bool(st.session_state.get("prefer_cc", False)),
                published_after=iso_timestamp_days_ago(past_days),
            )
            st.session_state["videos"] = vids
        if not vids:
            st.warning("No videos found.")
        else:
            st.success(f"Fetched {len(vids)} videos.")
            for v in vids[:8]:
                st.markdown(f"- [{v['title']}]({v['url']}) — {v['channel']} — {v['publishedAt']}")
            with st.spinner("Asking OpenAI to pick top 3 topics (YouTube + Google Trends)..."):
                trends = st.session_state.get("trends_data") or fetch_google_trends(
                    niche,
                    geo=st.session_state.get("trends_geo", "US"),
                    timeframe=st.session_state.get("trends_timeframe", "today 3-m"),
                )
                ranked_text = call_openai_rank_topics_combined(
                    niche,
                    vids,
                    trends,
                    days=int(st.session_state.get("virality_days", 7)),
                )
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

    len_col1, len_col2 = st.columns([1, 1])
    with len_col1:
        length_mode = st.selectbox("Length preset", ["Short (~60s)", "Custom (seconds)"])
    with len_col2:
        if length_mode == "Custom (seconds)":
            target_seconds = int(st.number_input("Target length (seconds)", min_value=10, max_value=600, value=90, step=5))
        else:
            target_seconds = 60
    st.session_state["target_duration_s"] = target_seconds

    # Safety checks
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
        for vid, lic in list(st.session_state.get("video_licenses", {}).items())[:6]:
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
        if not reuse:
            return llm_safe
        return llm_safe and any_cc

    # SEO metadata (AI only)
    st.subheader("SEO metadata")
    if st.button("Generate metadata (AI)"):
        try:
            with st.spinner("Generating SEO title/tags/description..."):
                meta = generate_metadata_openai(niche=chosen, trends={})
                st.session_state["seo_meta"] = meta
                st.success("SEO metadata ready.")
        except Exception as e:
            st.error(f"Metadata generation failed: {e}")

    if "seo_meta" in st.session_state:
        meta = st.session_state["seo_meta"]
        st.markdown("**Best Title**:")
        st.write(meta.get("best_title", ""))
        st.markdown("**Alternative Titles**:")
        for t in meta.get("alt_titles", [])[:3]:
            st.write(f"- {t}")
        st.markdown("**Tags**:")
        st.write(", ".join(meta.get("tags", [])))
        st.markdown("**Description**:")
        st.text_area("Description", meta.get("description", ""), height=180)
        st.caption(f"Predicted trend: {meta.get('predicted_trend', 'unknown')}. Reason: {meta.get('reasoning', '')}")
        use_title = st.checkbox("Use 'Best Title' for video title & thumbnail", value=True)
        if use_title:
            st.session_state["chosen_title"] = meta.get("best_title") or st.session_state.get("chosen_title")

    if st.button("Generate script for chosen topic"):
        with st.spinner("Generating script..."):
            script = generate_script_openai(st.session_state.get("chosen_title", chosen), target_seconds=st.session_state.get("target_duration_s"))
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
            allowed_to_build = True
            use_ai_images = bool(st.session_state.get("use_ai_imgs", False))
            if safety_mode and not passes_safety_gate():
                if "copyright_llm_status" not in st.session_state or "video_licenses" not in st.session_state:
                    try:
                        with st.spinner("Running safety checks..."):
                            context_text = "\n".join([f"{v['title']} — {v['url']}" for v in st.session_state.get("videos", [])])
                            status, explanation = llm_copyright_check(st.session_state.get("chosen_title") or "", context_text)
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
                    allowed_to_build = False
            if allowed_to_build:
                try:
                    with st.spinner("Preparing images and building video..."):
                        tmp_video = Path(tempfile.gettempdir()) / f"{safe_file_name(st.session_state.get('script','video'))}.mp4"
                        if use_ai_images:
                            tmpdir = Path(tempfile.mkdtemp())
                            ai_images = generate_cartoon_images_from_script(st.session_state["script"], tmpdir, max_images=min(int(num_images), 6))
                            images = [str(p) for p in ai_images]
                            if not images:
                                st.warning("AI image generation failed; falling back to Pexels.")
                                images = pexels_search_images(niche, per_page=int(num_images))
                        else:
                            images = pexels_search_images(niche, per_page=int(num_images))
                        video_path, srt_path = build_video_from_script_and_images(
                            st.session_state["script"],
                            images,
                            tmp_video,
                            title_text=st.session_state.get("chosen_title", "") or safe_file_name(niche),
                            target_duration_s=st.session_state.get("target_duration_s"),
                            strict_enforce=bool(strict_enforce),
                            motion_enabled=True,
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
                generate_thumbnail_pillow(safe_file_name(st.session_state.get("chosen_title", niche))[:80], tmp_thumb, subtitle="Auto-generated")
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

    if S3_BUCKET_NAME and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        st.write("S3 uploads:")
        if video_path and Path(video_path).exists():
            ok, url = s3_upload_file(Path(video_path), content_type="video/mp4")
            if ok: st.write(f"- Video: {url}")
        if audio_path and Path(audio_path).exists():
            ok, url = s3_upload_file(Path(audio_path), content_type="audio/mpeg")
            if ok: st.write(f"- Audio: {url}")
        if srt_path and Path(srt_path).exists():
            ok, url = s3_upload_file(Path(srt_path), content_type="application/x-subrip")
            if ok: st.write(f"- Subtitles: {url}")
        if thumb_path and Path(thumb_path).exists():
            ok, url = s3_upload_file(Path(thumb_path), content_type="image/jpeg")
            if ok: st.write(f"- Thumbnail: {url}")

st.caption(
    "Notes: This app generates SEO metadata with AI. MoviePy usage avoids editor/vfx modules for compatibility."
)


def ken_burns_frames(img: np.ndarray, width: int, height: int, duration: float, fps: int = 12,
                     zoom_start: float = 1.0, zoom_end: float = 1.06, pan_start: Tuple[float, float] = (0.5, 0.5), pan_end: Tuple[float, float] = (0.52, 0.48)) -> Tuple[List[np.ndarray], List[float]]:
    num_frames = max(1, int(round(duration * fps)))
    frames: List[np.ndarray] = []
    per_frame = duration / num_frames if num_frames > 0 else duration

    pil = Image.fromarray(img).convert("RGB")
    src_w, src_h = pil.size

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        scale = zoom_start + (zoom_end - zoom_start) * t
        cx = pan_start[0] + (pan_end[0] - pan_start[0]) * t
        cy = pan_start[1] + (pan_end[1] - pan_start[1]) * t

        crop_w = int(round(width / scale))
        crop_h = int(round(height / scale))
        crop_w = min(crop_w, src_w)
        crop_h = min(crop_h, src_h)

        left = int(round(cx * src_w - crop_w / 2))
        top = int(round(cy * src_h - crop_h / 2))
        left = max(0, min(left, src_w - crop_w))
        top = max(0, min(top, src_h - crop_h))
        right = left + crop_w
        bottom = top + crop_h

        patch = pil.crop((left, top, right, bottom)).resize((width, height), Image.LANCZOS)
        frames.append(np.array(patch))

    durations = [per_frame] * len(frames)
    return frames, durations