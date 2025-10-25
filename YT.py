import time
import os
import logging
import argparse
from datetime import datetime
import pytz
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
import openai
import obsws_python as obs

# Configuration
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OBS_PASSWORD = os.getenv('OBS_PASSWORD')
OBS_HOST = 'localhost'  # Adjust for remote OBS (e.g., ngrok URL)
OBS_PORT = 4455
OBS_TEXT_SOURCE_NAME = 'AI_Overlay'
CHANNEL_ID = os.getenv('YOUTUBE_CHANNEL_ID')
MODEL = 'gpt-4o'
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']

# Stream schedule (Pakistan Standard Time)
STREAM_SCHEDULE = {
    'Wednesday': '21:30',  # 9:30 PM PKT
    'Saturday': '21:30'
}
PKT = pytz.timezone('Asia/Karachi')

# Module contexts for AIMLDS course
MODULE_CONTEXTS = {
    1: "Python programming fundamentals (syntax, data structures, functions)",
    2: "Data preprocessing and visualization (Pandas, Matplotlib, Seaborn)",
    3: "Math for AI/ML/DS (linear algebra, calculus, probability, statistics)",
    4: "Evaluation parameters (Precision, Recall, F1, Bias/Variance tradeoff)",
    5: "Machine learning and deep learning (algorithms, neural networks, TensorFlow)",
    6: "Generative AI (APIs, MCP, tools like OpenAI, Hugging Face)",
    7: "Web frameworks (Flask, FastAPI for AI app development)",
    8: "SQL for data management and querying",
    9: "Containerization (Docker, Kubernetes for AI/ML deployments)",
    10: "Deployment (Git for version control, cloud platforms like AWS, Azure)",
    11: "No-code automation (tools like Zapier, Make for AI workflows)",
    12: "Rapid prototyping (building AI/ML prototypes quickly)"
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    filename='stream_log.txt',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize YouTube API client
def init_youtube_client():
    for attempt in range(3):
        try:
            flow = InstalledAppFlow.from_client_secrets_file('client_secrets.json', SCOPES)
            credentials = flow.run_local_server(port=0)
            return build('youtube', 'v3', credentials=credentials)
        except Exception as e:
            logging.error(f'OAuth attempt {attempt + 1} failed: {e}')
            time.sleep(5)
    logging.error('OAuth failed, falling back to API key')
    return build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Initialize OBS client
def init_obs_client():
    for attempt in range(3):
        try:
            client = obs.ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD, timeout=3)
            logging.info('Connected to OBS WebSocket')
            return client
        except Exception as e:
            logging.error(f'OBS connection attempt {attempt + 1} failed: {e}')
            time.sleep(5)
    logging.warning('OBS connection failed, using console fallback')
    return None

youtube = init_youtube_client()
openai.api_key = OPENAI_API_KEY
cl = init_obs_client()

# Check if current time is within stream schedule
def is_stream_time():
    now = datetime.now(PKT)
    day = now.strftime('%A')
    if day in STREAM_SCHEDULE:
        stream_time = datetime.strptime(STREAM_SCHEDULE[day], '%H:%M').time()
        current_time = now.time()
        stream_datetime = now.replace(hour=stream_time.hour, minute=stream_time.minute, second=0, microsecond=0)
        return stream_datetime <= now <= stream_datetime.replace(hour=stream_time.hour + 1)
    return False

# Get liveChatId
def get_live_chat_id(channel_id):
    try:
        request = youtube.search().list(
            part='snippet',
            channelId=channel_id,
            eventType='live',
            type='video'
        )
        response = request.execute()
        if response['items']:
            live_chat_id = response['items'][0]['snippet']['liveChatId']
            logging.info(f'Fetched liveChatId: {live_chat_id}')
            return live_chat_id
        return None
    except HttpError as e:
        logging.error(f'Error fetching liveChatId: {e}')
        return None

# Get live chat messages
def get_live_chat_messages(live_chat_id, next_page_token=None):
    try:
        request = youtube.liveChatMessages().list(
            liveChatId=live_chat_id,
            part='snippet,authorDetails',
            pageToken=next_page_token,
            maxResults=50
        )
        response = request.execute()
        logging.info(f'Fetched {len(response.get("items", []))} messages')
        return response
    except HttpError as e:
        logging.error(f'HTTP error {e.resp.status}: {e.content}')
        return None

# Post to YouTube chat
def post_to_chat(live_chat_id, message):
    try:
        youtube.liveChatMessages().insert(
            part='snippet',
            body={
                'snippet': {
                    'liveChatId': live_chat_id,
                    'type': 'textMessageEvent',
                    'textMessageDetails': {'messageText': message}
                }
            }
        ).execute()
        logging.info(f'Posted to chat: {message}')
    except HttpError as e:
        logging.error(f'Error posting to chat: {e}')

# Send question to LLM
def get_llm_response(question, context='', enable_humor=True):
    is_complex = any(keyword in question.lower() for keyword in ['debug', 'error', 'traceback', 'code', 'function'])
    humor_instruction = """
    Incorporate light, friendly humor (e.g., tech-related quips, pop culture references) to keep responses engaging, but keep it professional, inclusive, and relevant to the lecture context. Avoid sarcasm or humor that could be misinterpreted. If the question is complex or technical (e.g., about debugging or code errors), skip humor and provide a clear, serious answer.
    """ if enable_humor and not is_complex else "Provide a clear, concise, and professional answer without humor."

    prompt = f"""
    You are an AI Teaching Assistant for a YouTube live series on AI, Machine Learning, and Data Science.
    Your roles include:
    - FAQ Bot: Answer student questions concisely.
    - Summarizer: Provide key takeaways for the topic.
    - Coding Helper: Explain or generate code snippets with clear explanations.
    - Engagement: Encourage interaction with playful prompts if appropriate.
    {humor_instruction}
    Question: {question}
    Lecture context: {context}
    """
    try:
        response = openai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful and engaging teaching assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.8 if enable_humor and not is_complex else 0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f'OpenAI error: {e}')
        return "Sorry, I couldn't process that right now. My circuits need a quick reboot! 😅"

# Update OBS overlay
def update_obs_overlay(text):
    if cl:
        try:
            cl.set_input_settings(
                name=OBS_TEXT_SOURCE_NAME,
                settings={'text': text},
                overlay=True
            )
            logging.info(f'Updated OBS overlay: {text}')
        except Exception as e:
            logging.error(f'OBS error: {e}. Falling back to console.')
            print(f'Overlay text: {text}')
    else:
        logging.warning('OBS not connected. Outputting to console.')
        print(f'Overlay text: {text}')

# Main function
def main(live_chat_id, enable_humor, module):
    # Set lecture context based on module
    lecture_context = MODULE_CONTEXTS.get(module, "General AI/ML/DS topics")
    if module not in MODULE_CONTEXTS:
        logging.warning(f"Invalid module number: {module}. Using default context.")
        print(f"Warning: Module {module} not found. Using default context.")

    # Prompt for live_chat_id if not provided
    if not live_chat_id:
        live_chat_id = os.getenv('LIVE_CHAT_ID') or input("Enter the YouTube Live Chat ID for this session: ").strip()
        if not live_chat_id:
            logging.error("No live_chat_id provided.")
            print("Error: Live Chat ID is required.")
            return

    processed_messages = set()
    logging.info(f"Starting AI Teaching Assistant for liveChatId: {live_chat_id} (Humor: {'ON' if enable_humor else 'OFF'}, Module: {module}, Context: {lecture_context})")
    print(f"Starting AI Assistant for Chat ID: {live_chat_id} (Humor: {'ON' if enable_humor else 'OFF'}, Module: {module}, Context: {lecture_context})")
    
    while True:
        # Check if within stream time
        if not is_stream_time():
            logging.info('No stream scheduled now. Sleeping for 300 seconds.')
            print('No stream scheduled. Retrying in 5 minutes...')
            time.sleep(300)
            continue

        # Reinitialize clients if needed
        global youtube, cl
        if not youtube:
            youtube = init_youtube_client()
        if not cl:
            cl = init_obs_client()

        # Process chat messages
        response = get_live_chat_messages(live_chat_id)
        if response:
            for item in response.get('items', []):
                message_id = item['id']
                if message_id in processed_messages:
                    continue
                processed_messages.add(message_id)

                message_text = item['snippet']['displayMessage']
                author = item['authorDetails']['displayName']

                logging.info(f'New message from {author}: {message_text}')
                print(f'New message from {author}: {message_text}')

                if '?' in message_text or message_text.lower().startswith('q:'):
                    answer = get_llm_response(message_text, lecture_context, enable_humor)
                    chat_message = f'@{author}: {answer}'
                    post_to_chat(live_chat_id, chat_message)
                    overlay_text = f'Q from {author}: {message_text[:50]}...\nA: {answer}'
                    update_obs_overlay(overlay_text)

            next_page_token = response.get('nextPageToken')
            polling_interval = max(response['pollingIntervalMillis'] / 1000, 2) if response else 5
        else:
            polling_interval = 5

        time.sleep(polling_interval)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI Teaching Assistant for YouTube Live AIMLDS Course')
    parser.add_argument('--live-chat-id', type=str, default=os.getenv('LIVE_CHAT_ID'), help='YouTube Live Chat ID (or set LIVE_CHAT_ID env var)')
    parser.add_argument('--enable-humor', action='store_true', default=True, help='Enable humorous responses')
    parser.add_argument('--no-humor', action='store_false', dest='enable_humor', help='Disable humorous responses')
    parser.add_argument('--module', type=int, default=1, choices=range(1, 13), help='Course module number (1-12)')
    args = parser.parse_args()

    # Validate environment variables
    if not all([YOUTUBE_API_KEY, OPENAI_API_KEY, CHANNEL_ID]):
        logging.error('Missing environment variables.')
        print('Error: Set YOUTUBE_API_KEY, OPENAI_API_KEY, YOUTUBE_CHANNEL_ID.')
        exit(1)

    main(args.live_chat_id, args.enable_humor, args.module)