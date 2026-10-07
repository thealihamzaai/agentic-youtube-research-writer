# Agentic YouTube Research Writer

A multi-agent AI application that researches a specific YouTube channel and turns source-linked findings into a structured Markdown article.

The project uses **CrewAI** to coordinate a research agent and a writing agent, **OpenAI** models for reasoning and generation, **YouTube channel search** for topic discovery, and a **yt-dlp + transcript fallback** for recent videos that may not yet be available through semantic search.

## What it does

1. Accepts a YouTube channel `/videos` URL and a topic.
2. Searches the selected channel for relevant videos.
3. Expands common search terms when the first query produces weak results.
4. Falls back to a live scan of recent channel uploads using `yt-dlp`.
5. Pulls timestamped transcript snippets when available.
6. Generates a source-linked research brief.
7. Uses a second AI agent to turn that brief into a Markdown article.
8. Displays and exports both outputs through a Streamlit interface.

## Architecture

```text
User topic + YouTube channel
          |
          v
 YouTube Research Agent
          |
          +--> Semantic channel search
          |        |
          |        +--> Query expansion
          |        |
          |        +--> yt-dlp live fallback
          |                  |
          |                  +--> Transcript snippets
          |
          v
   Research Brief
          |
          v
 Technical Writing Agent
          |
          v
  Markdown Blog Article
```

## Tech stack

- Python
- CrewAI
- OpenAI models through CrewAI/LiteLLM
- Streamlit
- CrewAI YouTube search tools
- yt-dlp
- youtube-transcript-api
- python-dotenv

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4o-mini
YOUTUBE_CHANNEL_VIDEOS_URL=https://www.youtube.com/@codebasics/videos
```

Never commit your real `.env` file or API keys to GitHub.

### 4. Run the app

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

## Usage

- Paste the full YouTube channel `/videos` URL.
- Enter a topic to research.
- Choose the model and generation detail level.
- Run the workflow.
- Review the research brief and generated article before publishing.

## Notes and limitations

- The project depends on third-party APIs and YouTube page/transcript availability, so results can vary over time.
- Transcript access is not available for every video.
- The fallback scans recent uploads and is not intended to be a complete historical index of a channel.
- AI-generated content can contain mistakes. Verify important claims, quotations, timestamps, and source links before publication.
- Respect YouTube's terms, creator rights, and applicable copyright requirements when using generated content.

## Project status

This is a portfolio/demo project showing an applied multi-agent workflow for source-aware content research and writing. It is not presented as a production publishing system.
