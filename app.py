"""
Exotel to Slack Complete System
Automated call recording transcription and Slack posting with Zapier integration
SMART AGENT DETECTION: Matches from_number OR to_number against agent database
"""

import os
import json
import logging
import asyncio
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import requests
import sqlite3
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Exotel-Slack Complete System",
    description="Automated call transcription and Slack posting with smart agent detection",
    version="2.0.0"
)

# Configuration from environment
SLACK_WEBHOOK_URL = os.environ.get('SLACK_WEBHOOK_URL')
SLACK_BOT_TOKEN = os.environ.get('SLACK_BOT_TOKEN')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')
EXOTEL_API_KEY = os.environ.get('EXOTEL_API_KEY')
EXOTEL_API_TOKEN = os.environ.get('EXOTEL_API_TOKEN')
EXOTEL_SID = os.environ.get('EXOTEL_SID')
DATABASE_PATH = os.environ.get('DATABASE_PATH', 'processed_calls.db')

# Support agent phone number for direction detection
SUPPORT_NUMBER = os.environ.get('SUPPORT_NUMBER', '09631084471')

# Rate limiting configuration
PROCESSING_DELAY = int(os.environ.get('PROCESSING_DELAY', '5'))
MAX_CONCURRENT_CALLS = int(os.environ.get('MAX_CONCURRENT_CALLS', '3'))

# Department filtering configuration
ALLOWED_DEPARTMENTS = os.environ.get('ALLOWED_DEPARTMENTS', 'CUSTOMER SUPPORT')
ALLOWED_DEPT_LIST = [d.strip() for d in ALLOWED_DEPARTMENTS.split(',')] if ALLOWED_DEPARTMENTS.upper() != 'ALL' else []

# Processing semaphore (use threading.Semaphore for cross-thread compatibility)
processing_semaphore = threading.Semaphore(MAX_CONCURRENT_CALLS)

# Agent mapping - load from file
AGENT_MAPPING = {}
def load_agent_mapping():
    """Load agent mapping from JSON file"""
    global AGENT_MAPPING
    try:
        agent_file = Path('agent_mapping.json')
        if agent_file.exists():
            with open(agent_file, 'r') as f:
                data = json.load(f)
                AGENT_MAPPING = {k: v for k, v in data.items() if not k.startswith('_')}
            logger.info(f"✅ Loaded {len(AGENT_MAPPING)} agent mappings from database")
        else:
            logger.warning("agent_mapping.json not found - using default mappings")
    except Exception as e:
        logger.error(f"Failed to load agent mapping: {e}")

load_agent_mapping()

# Pydantic Models
class ZapierWebhookPayload(BaseModel):
    """Payload from Zapier webhook"""
    call_id: str = Field(..., description="Exotel Call ID (Sid)")
    from_number: str = Field(..., description="Caller number")
    to_number: str = Field(..., description="Called number")
    duration: int = Field(..., description="Call duration in seconds")
    recording_url: Optional[str] = Field(None, description="Recording URL from Exotel")
    timestamp: Optional[str] = Field(None, description="Call timestamp")
    status: str = Field(default="completed", description="Call status")
    agent_phone: Optional[str] = Field(None, description="Agent phone number")
    agent_name: Optional[str] = Field(None, description="Agent name")
    agent_slack_handle: Optional[str] = Field(None, description="Agent Slack handle")
    department: str = Field(default="Customer Success", description="Department")
    customer_segment: str = Field(default="General", description="Customer segment")


class WebhookResponse(BaseModel):
    """Response to Zapier webhook"""
    success: bool
    message: str
    call_id: str
    timestamp: str


# Database Manager
class DatabaseManager:
    """Manage SQLite database for duplicate prevention"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_calls (
                    call_id TEXT PRIMARY KEY,
                    from_number TEXT,
                    to_number TEXT,
                    duration INTEGER,
                    timestamp TEXT,
                    processed_at TEXT,
                    transcription_text TEXT,
                    slack_posted BOOLEAN,
                    status TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON processed_calls(timestamp)
            """)
            conn.commit()
        logger.info(f"Database initialized at {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def is_call_processed(self, call_id: str) -> bool:
        """Check if call has been processed"""
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT call_id FROM processed_calls WHERE call_id = ?",
                (call_id,)
            ).fetchone()
            return result is not None
    
    def mark_call_processing(self, call_id: str, call_data: Dict[str, Any]):
        """Mark call as being processed (prevents race condition)"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO processed_calls 
                (call_id, from_number, to_number, duration, timestamp, processed_at, 
                 transcription_text, slack_posted, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                call_id,
                call_data['from_number'],
                call_data['to_number'],
                call_data['duration'],
                call_data.get('timestamp', datetime.utcnow().isoformat()),
                datetime.utcnow().isoformat(),
                'Processing...',
                False,
                'processing'
            ))
            conn.commit()
        logger.info(f"Marked call {call_id} as processing (locked)")
    
    def mark_call_processed(self, call_data: Dict[str, Any], transcription: str, success: bool):
        """Mark call as processed"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO processed_calls 
                (call_id, from_number, to_number, duration, timestamp, processed_at, 
                 transcription_text, slack_posted, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                call_data['call_id'],
                call_data['from_number'],
                call_data['to_number'],
                call_data['duration'],
                call_data.get('timestamp', datetime.utcnow().isoformat()),
                datetime.utcnow().isoformat(),
                transcription,
                success,
                'completed' if success else 'failed'
            ))
            conn.commit()
        logger.info(f"Marked call {call_data['call_id']} as processed")
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics"""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM processed_calls").fetchone()[0]
            posted = conn.execute(
                "SELECT COUNT(*) FROM processed_calls WHERE slack_posted = 1"
            ).fetchone()[0]
            failed = conn.execute(
                "SELECT COUNT(*) FROM processed_calls WHERE status = 'failed'"
            ).fetchone()[0]
            
            return {
                'total_processed': total,
                'successfully_posted': posted,
                'failed': failed
            }


# Initialize database
db_manager = DatabaseManager(DATABASE_PATH)


# Transcription Service using OpenAI Whisper
class TranscriptionService:
    """Handle audio transcription using OpenAI Whisper"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.openai.com/v1/audio/transcriptions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def download_recording(self, recording_url: str, call_id: str) -> str:
        """Download recording from Exotel"""
        try:
            downloads_dir = Path("downloads")
            downloads_dir.mkdir(exist_ok=True)
            
            file_path = downloads_dir / f"{call_id}.mp3"
            
            auth = (EXOTEL_API_KEY, EXOTEL_API_TOKEN) if EXOTEL_API_KEY else None
            
            response = requests.get(recording_url, auth=auth, timeout=60)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded recording to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to download recording: {e}")
            raise
    
    def transcribe_audio(self, audio_file_path: str) -> str:
        """Transcribe audio file using OpenAI Whisper"""
        try:
            logger.info("Transcribing audio with OpenAI Whisper...")
            
            with open(audio_file_path, 'rb') as audio_file:
                files = {
                    'file': (os.path.basename(audio_file_path), audio_file, 'audio/mpeg')
                }
                data = {
                    'model': 'whisper-1',
                    'language': 'en',
                    'response_format': 'json'
                }
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    files=files,
                    data=data,
                    timeout=120
                )
            
            response.raise_for_status()
            result = response.json()
            
            transcription = result.get('text', '')
            
            if transcription:
                logger.info(f"✅ Transcription completed: {len(transcription)} characters")
                return transcription
            
            raise Exception("No transcription text returned from OpenAI Whisper")
            
        except Exception as e:
            logger.error(f"❌ OpenAI Whisper transcription error: {e}")
            raise
        finally:
            try:
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
                    logger.info(f"Cleaned up audio file: {audio_file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup audio file: {e}")


# MOM Generator Service
class MOMGenerator:
    """Generate Meeting Minutes from transcription using OpenAI"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_mom(self, transcription: str) -> str:
        """Generate structured Meeting Minutes from transcription with retry logic"""
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"Retry attempt {attempt + 1}/{max_retries} after {delay}s delay...")
                    time.sleep(delay)
                
                logger.info(f"Generating MOM from transcription (attempt {attempt + 1}/{max_retries})...")
                
                prompt = f"""You are an expert at creating concise Meeting Minutes (MOM) for customer support calls.

Analyze this call transcription and create a structured MOM with these sections:

**Tone:** [REQUIRED - Choose ONE: "Normal" or "Escalation"]
Determine if the call is an escalation (customer is frustrated/angry/upset/demanding manager) or normal (regular inquiry/polite conversation).

**Mood Analysis:** [REQUIRED - Choose ONE: "Satisfied", "Neutral", "Frustrated", "Angry", "Confused", "Relieved"]
Analyze the customer's emotional state throughout the call and overall satisfaction level.

**Concern Type:** [REQUIRED - Choose ONE or MORE from: Case Status Update, Insufficiency, Recharge, Product, Miscellaneous, Employment Verification, Document Issue, Payment Issue, Technical Problem]
Categorize the main concern or issue type discussed. Can be multiple types separated by commas.

**Issue Type:** [REQUIRED - Choose ONE or MORE from: Case Status Update, Insufficiency, Recharge, Product, Miscellaneous]
Categorize the main topics discussed. Can be multiple types separated by commas.

**Customer Issue:**
[Summarize the main problem/concern the customer is reporting - minimum 1 line]

**Key Discussion Points:**
[Extract ACTUAL QUOTES and statements from the real conversation - DO NOT rephrase or paraphrase]
[Use direct quotes from what was actually said in the call]
[Example: "Customer mentioned: 'I'm having trouble with...'" or "Agent explained: 'You need to...'"]
[List at least 2-3 actual discussion points from the conversation - each on a new line]

**Action Items:**
[List specific actions needed to resolve this issue - each action on a new line]

**Resolution Status:**
[State if issue was resolved or pending]

CRITICAL REQUIREMENTS: 
- The MOM must have AT LEAST 3 substantial lines of content
- For "Key Discussion Points" - USE ACTUAL WORDS from the conversation, NOT rephrased summaries
- Include direct quotes or verbatim statements whenever possible
- Each section should be detailed and informative
- Include specific details from the transcription
- MUST include Tone, Mood Analysis, and Concern Type at the beginning
- Mood Analysis is MANDATORY for customer satisfaction tracking

FORMAT:
**Tone:** [Normal/Escalation]
**Mood Analysis:** [Satisfied/Neutral/Frustrated/Angry/Confused/Relieved]
**Concern Type:** [Case Status Update/Insufficiency/Recharge/Product/Miscellaneous/Employment Verification/Document Issue/Payment Issue/Technical Problem]
**Issue Type:** [One or more categories]

**Customer Issue:**
[Summary]

**Key Discussion Points:**
[Actual quotes]

**Action Items:**
[Actions]

**Resolution Status:**
[Status]

Transcription:
{transcription}

Create the MOM in a clear, professional format with actual conversation content."""

                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a professional customer support analyst who creates detailed, well-structured meeting minutes with sufficient context. ALWAYS include Tone, Mood Analysis, and Concern Type fields in your response. These are critical for tracking customer satisfaction and issue categorization."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 800
                }
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                response.raise_for_status()
                result = response.json()
                
                if 'choices' in result and len(result['choices']) > 0:
                    mom = result['choices'][0]['message']['content'].strip()
                    logger.info(f"✅ MOM generated successfully: {len(mom)} characters")
                    return mom
                
                raise Exception("No MOM generated from OpenAI")
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    logger.warning(f"⚠️ OpenAI rate limit hit (429) - attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        logger.error("❌ Max retries reached for OpenAI rate limit")
                elif e.response.status_code == 401:
                    logger.error("❌ OpenAI API authentication failed - check API key")
                    break
                else:
                    logger.error(f"❌ OpenAI API HTTP error: {e.response.status_code}")
                    if attempt < max_retries - 1:
                        continue
                    break
            except Exception as e:
                logger.error(f"❌ MOM generation error: {e}")
                if attempt < max_retries - 1:
                    continue
                break
        
        logger.error("OpenAI API unavailable after all retries. Using structured fallback.")
        logger.warning("Falling back to structured transcription with 3+ lines")
        
        fallback_mom = f"""**Tone:** Normal
**Mood Analysis:** Neutral
**Concern Type:** Miscellaneous
**Issue Type:** Miscellaneous

**Call Summary:**
The call was regarding a customer support inquiry. The conversation lasted {len(transcription)} characters.

**Transcription:**
{transcription[:400] if len(transcription) > 400 else transcription}

**Note:** Automated MOM generation temporarily unavailable (OpenAI rate limit). This is a direct transcription of the call.

**Action Required:**
• Review call transcription for customer concerns
• Follow up with customer if needed
• Update ticket with call details"""
        
        return fallback_mom


# Google Drive Integration Service
class GoogleDriveService:
    """Upload transcript files to Google Drive for automatic NotebookLM sync"""
    
    def __init__(self, client_id: str = None, client_secret: str = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.service = None
        self.folder_id = None
        
    def setup_drive_service(self):
        """Initialize Google Drive API service"""
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            import pickle
            import os
            
            SCOPES = ['https://www.googleapis.com/auth/drive.file']
            
            creds = None
            # Check if token.pickle exists
            if os.path.exists('token.pickle'):
                with open('token.pickle', 'rb') as token:
                    creds = pickle.load(token)
            
            # If no valid credentials, get new ones
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if self.client_id and self.client_secret:
                        from google_auth_oauthlib.flow import Flow
                        flow = Flow.from_client_config(
                            {
                                "web": {
                                    "client_id": self.client_id,
                                    "client_secret": self.client_secret,
                                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                                    "token_uri": "https://oauth2.googleapis.com/token",
                                    "redirect_uris": ["http://localhost:8080"]
                                }
                            },
                            SCOPES
                        )
                        # For server deployment, we'll use a different approach
                        logger.warning("Google Drive OAuth requires manual setup - using local file fallback")
                        return False
                    else:
                        logger.warning("Google Drive credentials not found - file upload disabled")
                        return False
                
                # Save credentials for next run
                with open('token.pickle', 'wb') as token:
                    pickle.dump(creds, token)
            
            self.service = build('drive', 'v3', credentials=creds)
            
            # Create or find the transcripts folder
            self.folder_id = self._create_or_find_folder()
            return True
            
        except ImportError:
            logger.warning("Google Drive libraries not installed - file upload disabled")
            return False
        except Exception as e:
            logger.error(f"Failed to setup Google Drive service: {e}")
            return False
    
    def _create_or_find_folder(self):
        """Create or find the Exotel Transcripts folder in Google Drive"""
        try:
            # Search for existing folder
            results = self.service.files().list(
                q="name='Exotel Transcripts' and mimeType='application/vnd.google-apps.folder'",
                fields="files(id, name)"
            ).execute()
            
            folders = results.get('files', [])
            
            if folders:
                # Folder exists, return its ID
                folder_id = folders[0]['id']
                logger.info(f"Found existing 'Exotel Transcripts' folder: {folder_id}")
                return folder_id
            else:
                # Create new folder
                folder_metadata = {
                    'name': 'Exotel Transcripts',
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                
                folder = self.service.files().create(
                    body=folder_metadata,
                    fields='id'
                ).execute()
                
                folder_id = folder.get('id')
                logger.info(f"Created new 'Exotel Transcripts' folder: {folder_id}")
                return folder_id
                
        except Exception as e:
            logger.error(f"Failed to create/find folder: {e}")
            return None
    
    def send_transcript(self, transcript: str, call_data: Dict[str, Any]) -> bool:
        """Upload structured transcript file to Google Drive for automatic NotebookLM sync"""
        try:
            call_id = call_data.get('call_id', 'unknown')
            customer_number = call_data.get('from_number', 'unknown')
            agent_number = call_data.get('to_number', 'unknown')
            timestamp = call_data.get('start_time', 'unknown')
            duration = call_data.get('duration', 'unknown')
            
            # Create structured transcript content for NotebookLM
            structured_transcript = f"""CALL TRANSCRIPT - {call_id}
========================================

CALL METADATA:
- Call ID: {call_id}
- Customer Number: {customer_number}
- Agent Number: {agent_number}
- Date/Time: {timestamp}
- Duration: {duration}
- Source: Exotel Call Recording

TRANSCRIPT CONTENT:
{transcript}

========================================
END OF TRANSCRIPT - {call_id}
"""
            
            # Setup Google Drive service if not already done
            if not self.service:
                if not self.setup_drive_service():
                    logger.warning("Google Drive service not available - saving to local file instead")
                    return self._save_local_file(structured_transcript, call_id, timestamp)
            
            # Upload to Google Drive
            filename = f"transcript_{call_id}_{timestamp.replace(':', '-').replace(' ', '_')}.txt"
            
            # Create file metadata
            file_metadata = {
                'name': filename,
                'parents': [self.folder_id] if self.folder_id else []
            }
            
            # Create media upload
            from googleapiclient.http import MediaIoBaseUpload
            import io
            
            media = MediaIoBaseUpload(
                io.BytesIO(structured_transcript.encode('utf-8')),
                mimetype='text/plain',
                resumable=True
            )
            
            # Upload file
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            file_id = file.get('id')
            logger.info(f"📚 Transcript uploaded to Google Drive: {filename}")
            logger.info(f"📄 Transcript length: {len(transcript)} characters")
            logger.info(f"🔗 Google Drive file ID: {file_id}")
            logger.info(f"📁 File will automatically appear in NotebookLM when Google Drive is connected")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload transcript to Google Drive: {e}")
            # Fallback to local file
            logger.info("Falling back to local file storage...")
            return self._save_local_file(structured_transcript, call_id, timestamp)
    
    def _save_local_file(self, structured_transcript: str, call_id: str, timestamp: str) -> bool:
        """Fallback method to save transcript locally"""
        try:
            import os
            filename = f"transcript_{call_id}_{timestamp.replace(':', '-').replace(' ', '_')}.txt"
            filepath = f"transcripts/{filename}"
            
            # Create transcripts directory if it doesn't exist
            os.makedirs("transcripts", exist_ok=True)
            
            # Write transcript to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(structured_transcript)
            
            logger.info(f"📚 Transcript saved to local file: {filepath}")
            logger.info(f"📁 File ready for manual upload to NotebookLM: {filename}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save local transcript file: {e}")
            return False


# Slack User Lookup Service
class SlackUserLookup:
    """Query Slack workspace to find user info by phone number"""
    
    def __init__(self, bot_token: Optional[str]):
        self.bot_token = bot_token
        self.users_cache = {}
        self.cache_loaded = False
    
    def normalize_phone(self, phone: str) -> str:
        """Normalize phone number for comparison"""
        normalized = ''.join(filter(str.isdigit, phone))
        # Remove country code +91
        if normalized.startswith('91') and len(normalized) == 12:
            normalized = normalized[2:]
        # Remove leading 0
        if normalized.startswith('0') and len(normalized) == 11:
            normalized = normalized[1:]
        return normalized
    
    def load_users(self) -> bool:
        """Load all users from Slack workspace"""
        if not self.bot_token:
            logger.warning("Slack bot token not configured - cannot lookup users")
            return False
        
        try:
            logger.info("Loading users from Slack workspace...")
            
            response = requests.get(
                "https://slack.com/api/users.list",
                headers={"Authorization": f"Bearer {self.bot_token}"},
                timeout=10
            )
            
            response.raise_for_status()
            data = response.json()
            
            if not data.get('ok'):
                logger.error(f"Slack API error: {data.get('error')}")
                return False
            
            for member in data.get('members', []):
                if member.get('deleted') or member.get('is_bot'):
                    continue
                
                profile = member.get('profile', {})
                phone = profile.get('phone', '')
                
                if phone:
                    normalized_phone = self.normalize_phone(phone)
                    self.users_cache[normalized_phone] = {
                        'name': profile.get('real_name', profile.get('display_name', 'Unknown')),
                        'slack_handle': member.get('name', 'unknown'),
                        'user_id': member.get('id', ''),  # SLACK USER ID FOR TAGGING
                        'email': profile.get('email', ''),
                        'title': profile.get('title', ''),
                        'department': profile.get('fields', {}).get('department', 'Customer Success')
                    }
            
            self.cache_loaded = True
            logger.info(f"✅ Loaded {len(self.users_cache)} users from Slack workspace")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Slack users: {e}")
            return False
    
    def get_user_by_phone(self, phone: str) -> Optional[Dict[str, str]]:
        """Look up user in Slack by phone number"""
        if not self.cache_loaded:
            self.load_users()
        
        normalized = self.normalize_phone(phone)
        user = self.users_cache.get(normalized)
        
        if user:
            logger.info(f"📧 Found Slack user: {user['name']} ({user['email']}) - Will tag as <@{user['user_id']}>")
        
        return user


slack_user_lookup = SlackUserLookup(SLACK_BOT_TOKEN) if SLACK_BOT_TOKEN else None


# Slack Formatter
class SlackFormatter:
    """Format call data for Slack posting with smart agent detection"""
    
    @staticmethod
    def normalize_phone(phone: str) -> str:
        """Normalize phone number for comparison"""
        normalized = phone.replace('+', '').replace('-', '').replace(' ', '').replace('(', '').replace(')', '')
        # Remove country code +91
        if normalized.startswith('91') and len(normalized) == 12:
            normalized = normalized[2:]
        # Remove leading 0
        if normalized.startswith('0') and len(normalized) == 11:
            normalized = normalized[1:]
        return normalized
    
    @staticmethod
    def find_agent_from_call(from_number: str, to_number: str) -> Optional[Dict[str, str]]:
        """
        SMART AGENT DETECTION:
        Check both from_number and to_number against agent database.
        Returns agent info if found, None otherwise.
        """
        from_clean = SlackFormatter.normalize_phone(from_number)
        to_clean = SlackFormatter.normalize_phone(to_number)
        
        logger.info(f"🔍 Checking for agent match:")
        logger.info(f"   From: {from_number} (normalized: {from_clean})")
        logger.info(f"   To: {to_number} (normalized: {to_clean})")
        
        # First, try Slack workspace lookup for both numbers
        if slack_user_lookup:
            # Check from_number in Slack
            slack_user = slack_user_lookup.get_user_by_phone(from_number)
            if slack_user:
                user_id = slack_user.get('user_id', '')
                email = slack_user.get('email', '')
                slack_mention = f"<@{user_id}>" if user_id else f"({email})"
                
                logger.info(f"✅ Found agent (from_number) in Slack: {slack_user['name']} - {email}")
                return {
                    "phone": from_number,
                    "name": slack_user['name'],
                    "slack_mention": slack_mention,
                    "email": email,
                    "user_id": user_id,
                    "department": slack_user.get('department', 'Customer Success'),
                    "team": "Support",
                    "direction": "outgoing"  # Agent is caller
                }
            
            # Check to_number in Slack
            slack_user = slack_user_lookup.get_user_by_phone(to_number)
            if slack_user:
                user_id = slack_user.get('user_id', '')
                email = slack_user.get('email', '')
                slack_mention = f"<@{user_id}>" if user_id else f"({email})"
                
                logger.info(f"✅ Found agent (to_number) in Slack: {slack_user['name']} - {email}")
                return {
                    "phone": to_number,
                    "name": slack_user['name'],
                    "slack_mention": slack_mention,
                    "email": email,
                    "user_id": user_id,
                    "department": slack_user.get('department', 'Customer Success'),
                    "team": "Support",
                    "direction": "incoming"  # Agent is receiver
                }
        
        # Check agent_mapping.json database (97 agents)
        for mapped_phone, agent_data in AGENT_MAPPING.items():
            mapped_clean = SlackFormatter.normalize_phone(mapped_phone)
            
            # Check if from_number matches
            if mapped_clean == from_clean:
                email = agent_data.get('email', '')
                slack_mention = f"<@{email}>" if email else "@support"
                
                logger.info(f"✅ Found agent (from_number) in database: {agent_data.get('name')} - {email}")
                return {
                    "phone": from_number,
                    "name": agent_data.get('name', 'Support Agent'),
                    "slack_mention": slack_mention,
                    "email": email,
                    "user_id": "",
                    "department": agent_data.get('department', 'Customer Success'),
                    "team": agent_data.get('team', 'Support'),
                    "direction": "outgoing"  # Agent is caller
                }
            
            # Check if to_number matches
            if mapped_clean == to_clean:
                email = agent_data.get('email', '')
                slack_mention = f"<@{email}>" if email else "@support"
                
                logger.info(f"✅ Found agent (to_number) in database: {agent_data.get('name')} - {email}")
                return {
                    "phone": to_number,
                    "name": agent_data.get('name', 'Support Agent'),
                    "slack_mention": slack_mention,
                    "email": email,
                    "user_id": "",
                    "department": agent_data.get('department', 'Customer Success'),
                    "team": agent_data.get('team', 'Support'),
                    "direction": "incoming"  # Agent is receiver
                }
        
        # No agent found - return None
        logger.warning(f"⚠️ No agent found for from: {from_number} or to: {to_number}")
        return None
    
    @staticmethod
    def format_message(call_data: Dict[str, Any], mom: str, transcript: str = "") -> Dict[str, Any]:
        """Format Slack message with smart agent detection and email tagging
        Returns dict with 'message', 'customer_number', 'agent_name', 'timestamp'
        """
        
        # SMART DETECTION: Find which number belongs to support agent
        agent_info = SlackFormatter.find_agent_from_call(
            call_data['from_number'],
            call_data['to_number']
        )
        
        if agent_info:
            # Agent found in database
            agent_phone = agent_info['phone']
            agent_name = agent_info['name']
            agent_mention = agent_info['slack_mention']
            agent_email = agent_info.get('email', '')
            department = agent_info['department']
            direction = agent_info['direction']
            
            # Determine support vs customer number based on which one is the agent
            if direction == "outgoing":
                support_number = call_data['from_number']
                customer_number = call_data['to_number']
            else:
                support_number = call_data['to_number']
                customer_number = call_data['from_number']
            
            logger.info(f"📊 Call Summary: Agent={agent_name}, Direction={direction}")
        else:
            # No agent found - use fallback
            logger.warning(f"⚠️ No agent found in database for call {call_data['call_id']}")
            logger.warning(f"   From: {call_data['from_number']}")
            logger.warning(f"   To: {call_data['to_number']}")
            logger.warning(f"   💡 TIP: Add these numbers to agent_mapping.json if they are agents")
            
            # Assume incoming call (customer called support) as default
            support_number = call_data.get('to_number', 'Unknown')
            customer_number = call_data.get('from_number', 'Unknown')
            agent_name = call_data.get('agent_name', 'Unknown Agent')
            agent_mention = '@support'
            agent_email = call_data.get('agent_email', '')
            department = call_data.get('department', 'Customer Success')
            direction = "incoming"  # Default assumption
        
        # Format timestamp - CONVERT TO IST
        timestamp = call_data.get('timestamp', datetime.utcnow().isoformat())
        if 'T' in timestamp:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            ist_offset = timedelta(hours=5, minutes=30)
            dt_ist = dt + ist_offset
            timestamp_formatted = dt_ist.strftime('%Y-%m-%d %H:%M:%S IST')
            date_only = dt_ist.strftime('%Y-%m-%d')
        else:
            timestamp_formatted = timestamp + ' IST'
            date_only = timestamp
        
        duration_sec = call_data.get('duration', 0)
        duration_formatted = f"{duration_sec}s ({int(duration_sec // 60)}m {duration_sec % 60}s)"
        
        # Customer Legal Name (use phone number for now, can be enhanced with Google Sheets lookup)
        customer_legal_name = f"Customer {customer_number}"
        
        # Build Exotel recording link
        exotel_link = call_data.get('recording_url', 'N/A')
        
        # Transcript section removed - transcripts now go to NotebookLM only
        
        # Main message body - clean and focused
        message = f"""📞 *Customer Support Call Summary*

━━━━━━━━━━━━━━━━━━━━━━━━━━

👤 *Customer:* {customer_legal_name}
📱 *Customer Number:* `{customer_number}`

👔 *Agent:* {agent_name} {agent_mention}
🏢 *Department:* {department}

📅 *Date of Call:* {date_only}
🕐 *Time:* {timestamp_formatted}
⏱️ *Duration:* {duration_formatted}

━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 *Meeting Minutes (MOM):*

{mom}

━━━━━━━━━━━━━━━━━━━━━━━━━━

🔗 *Call Recording:* <{exotel_link}|Listen on Exotel>
🆔 *Call ID:* `{call_data['call_id']}`"""
        
        return {
            'message': message,
            'customer_number': customer_number,
            'customer_legal_name': customer_legal_name,
            'agent_name': agent_name,
            'timestamp': timestamp_formatted,
            'call_id': call_data['call_id'],
            'exotel_link': exotel_link,
            'mom': mom  # Store MOM separately for potential future use
        }
    
    @staticmethod
    def post_to_slack(message_data: Dict[str, Any], webhook_url: str) -> bool:
        """Post formatted message to Slack
        
        Args:
            message_data: Dict containing 'message' and other metadata
            webhook_url: Slack webhook URL
        """
        try:
            main_message = message_data['message']
            
            # Post message via webhook
            payload = {
                "text": main_message,
                "mrkdwn": True,
                "unfurl_links": False,
                "unfurl_media": False
            }
            
            response = requests.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            response.raise_for_status()
            
            logger.info("✅ Successfully posted message to Slack")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to post to Slack: {e}")
            return False


# Initialize services
transcription_service = TranscriptionService(OPENAI_API_KEY) if OPENAI_API_KEY else None
mom_generator = MOMGenerator(OPENAI_API_KEY) if OPENAI_API_KEY else None
google_drive_service = GoogleDriveService(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET) if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET else None


# Background processing with threading semaphore
async def process_call_with_rate_limit(call_data: Dict[str, Any]):
    """Process call with rate limiting to prevent server overload"""
    call_id = call_data['call_id']
    
    # Acquire semaphore (wait if max concurrent calls reached)
    processing_semaphore.acquire()
    try:
        logger.info(f"[Queue] Starting processing for call {call_id}")
        
        # Check if recording URL is provided
        recording_url = call_data.get('recording_url')
        if not recording_url:
            logger.warning(f"No recording URL for call {call_id}")
            db_manager.mark_call_processed(call_data, "No recording available", False)
            return
        
        # Download recording
        logger.info(f"Downloading recording for call {call_id}")
        audio_file = transcription_service.download_recording(recording_url, call_id)
        
        # Transcribe audio
        logger.info(f"Transcribing call {call_id}")
        transcription = transcription_service.transcribe_audio(audio_file)
        
        if not transcription or len(transcription.strip()) == 0:
            raise Exception("Transcription returned empty text")
        
        logger.info(f"Transcription completed: {len(transcription)} characters")
        
        # Upload transcript to Google Drive for automatic NotebookLM sync
        if google_drive_service:
            logger.info(f"Uploading transcript to Google Drive for call {call_id}")
            drive_success = google_drive_service.send_transcript(transcription, call_data)
            if drive_success:
                logger.info(f"✅ Transcript successfully uploaded to Google Drive")
            else:
                logger.warning(f"⚠️ Failed to upload transcript to Google Drive")
        else:
            logger.warning("Google Drive service not available - transcript not uploaded")
        
        # Generate MOM from transcription
        if mom_generator:
            logger.info(f"Generating MOM for call {call_id}")
            mom = mom_generator.generate_mom(transcription)
        else:
            logger.warning("MOM generator not available, using transcription")
            mom = transcription
        
        # Format and post to Slack (MOM only, no transcript)
        logger.info(f"Formatting message for Slack")
        slack_message_data = SlackFormatter.format_message(call_data, mom, "")
        
        logger.info(f"Posting to Slack")
        success = SlackFormatter.post_to_slack(
            message_data=slack_message_data,
            webhook_url=SLACK_WEBHOOK_URL
        )
        
        # Mark as processed
        db_manager.mark_call_processed(call_data, transcription, success)
        
        if success:
            logger.info(f"✅ Successfully completed processing for call {call_id}")
        else:
            logger.error(f"❌ Failed to post to Slack for call {call_id}")
        
        # Add delay before next call (rate limiting)
        if PROCESSING_DELAY > 0:
            logger.info(f"⏸️ Waiting {PROCESSING_DELAY} seconds before next call...")
            await asyncio.sleep(PROCESSING_DELAY)
            
    except Exception as e:
        logger.error(f"❌ Error processing call {call_id}: {e}")
        db_manager.mark_call_processed(call_data, f"Error: {str(e)}", False)
    finally:
        processing_semaphore.release()


def process_call_background(call_data: Dict[str, Any]):
    """Wrapper to run async processing in background"""
    try:
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(process_call_with_rate_limit(call_data))
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Error in background processing wrapper: {e}")


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Exotel-Slack Complete System",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Smart Agent Detection (97 agents)",
            "OpenAI Whisper Transcription",
            "AI MOM Generation",
            "Slack User Tagging by Phone & Email",
            "Duplicate Prevention",
            "IST Timezone Support"
        ],
        "agents_loaded": len(AGENT_MAPPING),
        "endpoints": {
            "health": "/health",
            "zapier_webhook": "/webhook/zapier",
            "stats": "/stats"
        },
        "documentation": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    stats = db_manager.get_stats()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "database": "connected",
        "agents_loaded": len(AGENT_MAPPING),
        "stats": stats,
        "services": {
            "transcription": "enabled (OpenAI Whisper)" if transcription_service else "disabled",
            "mom_generator": "enabled" if mom_generator else "disabled",
            "google_drive": "enabled" if google_drive_service else "disabled (set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)",
            "slack": "enabled" if SLACK_WEBHOOK_URL else "disabled",
            "slack_user_lookup": "enabled" if slack_user_lookup else "disabled (set SLACK_BOT_TOKEN)",
            "agent_database": f"{len(AGENT_MAPPING)} agents loaded"
        }
    }


@app.post("/webhook/zapier", response_model=WebhookResponse)
async def zapier_webhook(
    payload: ZapierWebhookPayload,
    background_tasks: BackgroundTasks
):
    """Main webhook endpoint for Zapier integration"""
    try:
        call_id = payload.call_id
        logger.info(f"📞 Received webhook for call {call_id}")
        logger.info(f"   From: {payload.from_number}")
        logger.info(f"   To: {payload.to_number}")
        
        if db_manager.is_call_processed(call_id):
            logger.info(f"Duplicate call detected: {call_id}")
            return WebhookResponse(
                success=True,
                message="Duplicate call - already processed",
                call_id=call_id,
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
        
        if not SLACK_WEBHOOK_URL:
            raise HTTPException(status_code=500, detail="Slack webhook not configured")
        
        if not transcription_service:
            raise HTTPException(status_code=500, detail="Transcription service not configured")
        
        # DEPARTMENT FILTER: Check if agent's department is allowed
        if ALLOWED_DEPT_LIST:
            # Detect agent from call numbers
            agent_info = SlackFormatter.find_agent_from_call(
                payload.from_number,
                payload.to_number
            )
            
            if agent_info:
                agent_dept = agent_info.get('department', 'Unknown')
                if agent_dept not in ALLOWED_DEPT_LIST:
                    logger.info(f"🚫 Skipping call {call_id} - Agent department '{agent_dept}' not in allowed list")
                    logger.info(f"   Allowed departments: {', '.join(ALLOWED_DEPT_LIST)}")
                    logger.info(f"   Agent: {agent_info.get('name', 'Unknown')}")
                    return WebhookResponse(
                        success=True,
                        message=f"Call skipped - Department '{agent_dept}' not in filter (allowed: {', '.join(ALLOWED_DEPT_LIST)})",
                        call_id=call_id,
                        timestamp=datetime.utcnow().isoformat() + "Z"
                    )
                else:
                    logger.info(f"✅ Agent department '{agent_dept}' matches filter - processing call")
            else:
                logger.warning(f"⚠️ Could not detect agent for call {call_id} - skipping due to department filter")
                logger.info(f"   From: {payload.from_number}, To: {payload.to_number}")
                return WebhookResponse(
                    success=True,
                    message="Call skipped - Agent not found in database (department filter active)",
                    call_id=call_id,
                    timestamp=datetime.utcnow().isoformat() + "Z"
                )
        
        call_data = {
            'call_id': call_id,
            'from_number': payload.from_number,
            'to_number': payload.to_number,
            'duration': payload.duration,
            'recording_url': payload.recording_url,
            'timestamp': payload.timestamp or datetime.utcnow().isoformat(),
            'status': payload.status,
            'agent_phone': payload.agent_phone,
            'agent_name': payload.agent_name,
            'agent_slack_handle': payload.agent_slack_handle,
            'department': payload.department,
            'customer_segment': payload.customer_segment
        }
        
        db_manager.mark_call_processing(call_id, call_data)
        
        background_tasks.add_task(process_call_background, call_data)
        
        logger.info(f"✅ Queued processing for call {call_id}")
        
        return WebhookResponse(
            success=True,
            message="Call queued for processing",
            call_id=call_id,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in webhook handler: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get processing statistics"""
    stats = db_manager.get_stats()
    return {
        "stats": stats,
        "agents_loaded": len(AGENT_MAPPING),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/call/{call_id}")
async def get_call_details(call_id: str):
    """Get details of a processed call"""
    with db_manager._get_connection() as conn:
        result = conn.execute(
            "SELECT * FROM processed_calls WHERE call_id = ?",
            (call_id,)
        ).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="Call not found")
        
        return dict(result)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "details": str(exc),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("=" * 60)
    logger.info("Starting Exotel-Slack Complete System v2.0")
    logger.info("=" * 60)
    logger.info(f"Database: {DATABASE_PATH}")
    logger.info(f"Agent Database: {len(AGENT_MAPPING)} agents loaded")
    logger.info(f"Transcription: {'Enabled (OpenAI Whisper)' if transcription_service else 'Disabled'}")
    logger.info(f"MOM Generator: {'Enabled' if mom_generator else 'Disabled'}")
    logger.info(f"Google Drive: {'Enabled' if google_drive_service else 'Disabled (set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)'}")
    logger.info(f"Slack: {'Enabled' if SLACK_WEBHOOK_URL else 'Disabled'}")
    logger.info(f"Slack User Lookup: {'Enabled' if slack_user_lookup else 'Disabled (set SLACK_BOT_TOKEN)'}")
    logger.info(f"Support Number: {SUPPORT_NUMBER}")
    logger.info(f"Rate Limiting: {PROCESSING_DELAY}s delay, {MAX_CONCURRENT_CALLS} concurrent calls")
    logger.info(f"Smart Agent Detection: Enabled ✅")
    if ALLOWED_DEPT_LIST:
        logger.info(f"🔍 Department Filter: ENABLED - Only processing: {', '.join(ALLOWED_DEPT_LIST)}")
    else:
        logger.info(f"🔍 Department Filter: DISABLED - Processing all departments")
    logger.info("=" * 60)
    
    Path("downloads").mkdir(exist_ok=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
