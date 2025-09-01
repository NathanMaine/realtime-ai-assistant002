# Real-Time AI Meeting Assistant

## Overview
This is a Streamlit-based application that transcribes live audio from your microphone, summarizes meeting content, and extracts action items using the xAI Grok API.

## Features
- Records 5-second audio clips from your microphone.
- Transcribes audio using the Whisper model.
- Queries the xAI Grok API for summaries and action items.
- Provides text-to-speech feedback.

## Installation
1. Clone the repository: `git clone https://github.com/yourusername/realtime-ai-assistant.git`
2. Navigate: `cd realtime-ai-assistant`
3. Create venv: `python3 -m venv venv`
4. Activate: `source venv/bin/activate`
5. Install: `pip install -r requirements.txt`
6. Set API key: Create `.env` with `XAI_API_KEY=your-api-key`.

## Usage
- Run: `streamlit run app.py`
- Click "Record & Analyze" to record and see results.

## Recent Updates
- Added audio recording.
- Enhanced UI and error handling.

## Contributing
- Fork and create a feature branch.
- Commit and open a pull request.

## License
[MIT] - Consult legal.

## Additional Setup
See [docs/setup.md](docs/setup.md).
