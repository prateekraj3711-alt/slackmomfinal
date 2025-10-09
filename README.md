# Exotel-Slack Complete System

**Automated call recording transcription and Slack posting system with Zapier integration**

A production-ready system that automatically fetches Exotel call recordings, transcribes them using AssemblyAI, and posts formatted meeting minutes to Slack. Built to handle 50+ support agents with zero duplicate processing.

## ✨ Features

- ✅ **Zapier Integration**: Seamless webhook integration for automated workflows
- ✅ **Duplicate Prevention**: SQLite-based deduplication ensures no call is processed twice
- ✅ **Audio Transcription**: High-quality transcription using AssemblyAI with speaker detection
- ✅ **Exact Format**: Posts to Slack in the EXACT format you specified (matches image)
- ✅ **Call Direction Logic**: Automatically determines incoming vs outgoing calls
- ✅ **Scalable**: Designed to handle 50+ concurrent support agents
- ✅ **No AI Rephrasing**: Uses actual transcription text as MOM (Meeting Minutes)
- ✅ **Background Processing**: Instant webhook responses with async transcription
- ✅ **Production Ready**: Includes Docker, health checks, error handling, and logging

## 🏗️ Architecture

```
┌─────────┐      ┌────────┐      ┌──────────────┐      ┌─────────────────┐      ┌───────┐
│ Exotel  │─────▶│ Zapier │─────▶│ This System  │─────▶│  Transcription  │─────▶│ Slack │
│  Call   │      │ Trigger│      │   (FastAPI)  │      │   (AssemblyAI)  │      │       │
└─────────┘      └────────┘      └──────────────┘      └─────────────────┘      └───────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │   SQLite DB  │
                                   │  (Duplicates)│
                                   └──────────────┘
```

## 📋 Requirements

- Python 3.11+
- Slack Workspace with webhook
- AssemblyAI API key (for transcription)
- Exotel account with API credentials
- Zapier account (Free tier works, but Pro+ recommended for high volume)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd exotel-slack-complete-system
cp env.example .env
```

### 2. Configure Environment

Edit `.env` file with your credentials:

```env
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
ASSEMBLYAI_API_KEY=your_assemblyai_api_key
EXOTEL_API_KEY=your_exotel_api_key
EXOTEL_API_TOKEN=your_exotel_api_token
EXOTEL_SID=your_exotel_sid
SUPPORT_NUMBER=09631084471
```

### 3. Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The server will start at `http://localhost:8000`

### 4. Run with Docker

```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

## 🌐 Deployment Options

### Option A: Deploy to Render (Recommended - Free Tier Available)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin YOUR_REPO_URL
   git push -u origin main
   ```

2. **Deploy on Render**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect the `render.yaml` configuration
   - Add environment variables in Render dashboard
   - Click "Create Web Service"

3. **Get Your URL**
   ```
   https://your-app-name.onrender.com
   ```

### Option B: Deploy to Railway

1. Install Railway CLI: `npm install -g @railway/cli`
2. Login: `railway login`
3. Initialize: `railway init`
4. Deploy: `railway up`
5. Add environment variables: `railway variables`

### Option C: Deploy to AWS/GCP/Azure

Use the provided `Dockerfile` for containerized deployment.

## 📡 API Endpoints

### `POST /webhook/zapier`

Main webhook endpoint for Zapier integration.

**Request Body:**
```json
{
  "call_id": "CA1234567890abcdef",
  "from_number": "+919876543210",
  "to_number": "09631084471",
  "duration": 125,
  "recording_url": "https://exotel.com/recordings/xxx.mp3",
  "timestamp": "2025-10-09T17:58:27Z",
  "status": "completed",
  "agent_name": "Prateek Raj",
  "agent_slack_handle": "@prateek.raj",
  "department": "Customer Success",
  "customer_segment": "General"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Call queued for processing",
  "call_id": "CA1234567890abcdef",
  "timestamp": "2025-10-09T17:58:30Z"
}
```

### `GET /health`

Health check endpoint with statistics.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-09T17:58:30Z",
  "database": "connected",
  "stats": {
    "total_processed": 150,
    "successfully_posted": 148,
    "failed": 2
  },
  "services": {
    "transcription": "enabled",
    "slack": "enabled"
  }
}
```

### `GET /stats`

Get processing statistics.

### `GET /call/{call_id}`

Get details of a specific processed call.

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `SLACK_WEBHOOK_URL` | Yes | Slack incoming webhook URL | `https://hooks.slack.com/services/...` |
| `ASSEMBLYAI_API_KEY` | Yes | AssemblyAI API key for transcription | `abc123...` |
| `EXOTEL_API_KEY` | Yes | Exotel API key | `your_api_key` |
| `EXOTEL_API_TOKEN` | Yes | Exotel API token | `your_api_token` |
| `EXOTEL_SID` | Yes | Exotel account SID | `your_sid` |
| `SUPPORT_NUMBER` | Yes | Your support phone number | `09631084471` |
| `DATABASE_PATH` | No | Database file path | `processed_calls.db` |
| `PORT` | No | Server port | `8000` |

### Slack Message Format

The system posts messages in this EXACT format (matching your image):

```
📞 Support Number:
09631084471

📱 Candidate/Customer Number:
09631084471

❗ Concern:
Call inquiry: Person you are speaking with has put your call on ... (Tone: Neutral)

👤 CS Agent:
@prateek.raj <09631084471>

🏢 Department:
Customer Success

⏰ Timestamp:
2025-10-04 17:58:27 UTC

📋 Call Metadata:
• Call ID: `86d8a88279e4f1ad7417e6556c3e1984`
• Duration: 56s
• Status: Completed
• Agent: Prateek Raj
• Customer Segment: General

📝 Full Transcription:
```
[Complete transcription with speaker labels]
```

🎧 Recording/Voice Note:
Recording processed and transcribed above (as per no-audio requirement)
```

## 🔗 Zapier Integration

See [ZAPIER_INTEGRATION_GUIDE.md](ZAPIER_INTEGRATION_GUIDE.md) for complete setup instructions.

**Quick Overview:**

1. **Trigger**: Exotel → "New Call" or "Call Completed"
2. **Action**: Webhooks by Zapier → POST
3. **URL**: `https://your-app.onrender.com/webhook/zapier`
4. **Payload**: Map Exotel fields to JSON format above
5. **Test**: Run test call and verify Slack message

## 🎯 Call Direction Logic

The system automatically determines call direction:

**Incoming Call:**
- From = Customer/Candidate number
- To = Support number

**Outgoing Call:**
- From = Support number
- To = Customer/Candidate number

Logic: If `from_number` contains `SUPPORT_NUMBER`, it's outgoing; otherwise incoming.

## 🔒 Duplicate Prevention

The system uses SQLite database to track processed calls:

- Each call is identified by unique `call_id` (Exotel Sid)
- Before processing, checks if `call_id` exists in database
- If duplicate detected, returns success but skips processing
- No call is ever transcribed or posted twice

**Database Schema:**
```sql
CREATE TABLE processed_calls (
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
```

## 📊 Handling 50+ Agents

The system is designed to scale:

- **Async Processing**: Background tasks don't block webhook responses
- **Fast Responses**: Zapier receives instant acknowledgment (< 100ms)
- **Concurrent Processing**: Multiple calls processed simultaneously
- **Database Locking**: SQLite handles concurrent access safely
- **Error Isolation**: One failed call doesn't affect others

**Recommended Setup for High Volume:**
- Deploy on Render Standard plan or higher (not free tier)
- Use Zapier Professional+ for faster triggers (1-minute polling)
- Consider upgrading to PostgreSQL for very high volume (>1000 calls/day)

## 🧪 Testing

### Test Health Endpoint

```bash
curl https://your-app.onrender.com/health
```

### Test Webhook with Sample Data

```bash
curl -X POST https://your-app.onrender.com/webhook/zapier \
  -H "Content-Type: application/json" \
  -d '{
    "call_id": "TEST123456",
    "from_number": "+919876543210",
    "to_number": "09631084471",
    "duration": 60,
    "recording_url": "https://example.com/test.mp3",
    "agent_name": "Test Agent",
    "agent_slack_handle": "@test",
    "department": "Customer Success",
    "customer_segment": "General"
  }'
```

### Test with Actual Exotel Recording

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive testing procedures.

## 🐛 Troubleshooting

### Issue: Webhook not receiving calls

**Check:**
- Is Zapier Zap turned on?
- Is webhook URL correct in Zapier?
- Check Zapier Task History for errors

**Solution:**
- Verify URL: `curl https://your-app.onrender.com/health`
- Check Render logs for incoming requests
- Test webhook with curl command above

### Issue: Transcription fails

**Check:**
- Is `ASSEMBLYAI_API_KEY` set correctly?
- Is recording URL accessible?
- Check Render logs for transcription errors

**Solution:**
- Verify AssemblyAI API key
- Test recording URL manually
- Check Exotel credentials for recording download

### Issue: Slack messages not posting

**Check:**
- Is `SLACK_WEBHOOK_URL` set correctly?
- Test webhook URL with curl:
  ```bash
  curl -X POST YOUR_SLACK_WEBHOOK_URL \
    -H 'Content-Type: application/json' \
    -d '{"text":"Test message"}'
  ```

### Issue: Duplicates being processed

**Check:**
- Is database persisting between restarts?
- Check database file exists and is writable
- Verify `call_id` is unique in Exotel data

**Solution:**
- Ensure persistent storage is enabled (Render: use disk storage)
- Check file permissions
- Query database: `sqlite3 processed_calls.db "SELECT * FROM processed_calls"`

## 📖 Additional Documentation

- [ZAPIER_INTEGRATION_GUIDE.md](ZAPIER_INTEGRATION_GUIDE.md) - Complete Zapier setup
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Detailed deployment instructions
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API documentation
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Testing procedures

## 🤝 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review logs: `docker-compose logs -f` (Docker) or Render dashboard (cloud)
3. Check Zapier Task History
4. Verify all environment variables are set correctly

## 📝 License

MIT License - feel free to use and modify for your needs.

## 🎯 Features Checklist

- [x] Zapier webhook integration
- [x] Exotel recording download
- [x] AssemblyAI transcription with speaker labels
- [x] Duplicate prevention with SQLite
- [x] Exact Slack format matching image
- [x] Call direction logic (incoming/outgoing)
- [x] Background async processing
- [x] Handles 50+ agents
- [x] No AI rephrasing (actual transcription as MOM)
- [x] Docker support
- [x] Health checks and monitoring
- [x] Error handling and logging
- [x] Production-ready deployment

---

**Built with FastAPI, AssemblyAI, and Slack** 🚀

