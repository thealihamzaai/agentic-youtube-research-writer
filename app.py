"""Streamlit app for a CrewAI-powered YouTube research and writing workflow."""

import inspect
import os
import re
from typing import Any, Dict, List, Type

import streamlit as st
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from crewai_tools import YoutubeChannelSearchTool
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled, YouTubeTranscriptApi
from yt_dlp import YoutubeDL


st.set_page_config(
    page_title="Agentic YouTube Research Writer",
    page_icon="🎬",
    layout="wide",
)

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
DEFAULT_CHANNEL_VIDEOS_URL = os.getenv(
    "YOUTUBE_CHANNEL_VIDEOS_URL",
    "https://www.youtube.com/@codebasics/videos",
)

TOKEN_PRESETS: Dict[str, Dict[str, float]] = {
    "Economy": {"max_tokens": 512, "temperature": 0.20},
    "Balanced": {"max_tokens": 1024, "temperature": 0.30},
    "Detailed": {"max_tokens": 2048, "temperature": 0.35},
}

VIDEOS_URL_RE = re.compile(r"^https?://(www\.)?youtube\.com/[^?#]+/videos/?$", re.I)
HANDLE_EXTRACT_RE = re.compile(r"/@([^/]+)", re.I)
UCID_EXTRACT_RE = re.compile(r"/channel/(UC[0-9A-Za-z_-]{20,})", re.I)


def make_llm(model_name: str, preset_key: str) -> LLM:
    preset = TOKEN_PRESETS[preset_key]
    return LLM(
        model=f"openai/{model_name}",
        temperature=preset["temperature"],
        max_tokens=int(preset["max_tokens"]),
    )


def require_videos_url(value: str) -> str:
    value = (value or "").strip()
    if not VIDEOS_URL_RE.match(value):
        raise ValueError(
            "Paste the channel's full /videos URL, for example: "
            "https://www.youtube.com/@codebasics/videos"
        )
    return value


def strip_videos_tab(videos_url: str) -> str:
    return re.sub(r"/videos/?$", "", videos_url, flags=re.I)


def expand_query(query: str) -> List[str]:
    normalized = (query or "").strip()
    variants = [normalized]
    lower = normalized.lower()

    expansions = {
        "rag": [
            "retrieval augmented generation",
            "retrieval-augmented generation",
            "RAG pipeline",
            "RAG vs fine-tuning",
            "RAG with vector database",
        ],
        "llm": ["large language model", "large language models"],
        "llms": ["large language model", "large language models"],
    }
    variants.extend(expansions.get(lower, []))

    if " " in lower:
        variants.extend([lower.replace(" ", "-"), lower.replace("-", " ")])

    seen = set()
    unique = []
    for item in variants:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def looks_empty(output: Any) -> bool:
    if output is None:
        return True
    if isinstance(output, (list, tuple, dict)) and not output:
        return True
    if isinstance(output, str):
        text = output.strip().lower()
        return not text or "no relevant content found" in text
    return False


def format_timestamp(seconds: float) -> str:
    total = int(seconds or 0)
    return f"{total // 60:02d}:{total % 60:02d}"


def _transcript_segments(video_id: str) -> List[dict]:
    """Support both older and newer youtube-transcript-api interfaces."""
    try:
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            return YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

        fetched = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        segments = []
        for item in fetched:
            if hasattr(item, "text"):
                segments.append(
                    {
                        "text": item.text,
                        "start": getattr(item, "start", 0.0),
                    }
                )
            elif isinstance(item, dict):
                segments.append(item)
        return segments
    except (TranscriptsDisabled, NoTranscriptFound):
        return []
    except Exception:
        return []


def fetch_transcript_snippets(
    video_id: str,
    terms: List[str],
    max_hits: int = 3,
) -> List[str]:
    searchable_terms = [term.lower() for term in terms if term]
    hits: List[str] = []

    for segment in _transcript_segments(video_id):
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        if any(term in text.lower() for term in searchable_terms):
            hits.append(f"[{format_timestamp(segment.get('start', 0.0))}] {text}")
            if len(hits) >= max_hits:
                break
    return hits


def live_channel_search(
    channel_base_url: str,
    query: str,
    limit: int = 6,
    include_snippets: bool = True,
) -> str:
    """Fallback search over recent channel uploads using yt-dlp metadata."""
    ydl_options = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "playlistend": 100,
    }
    matches: List[str] = []
    query_variants = [item.lower() for item in expand_query(query)]

    with YoutubeDL(ydl_options) as ydl:
        info = ydl.extract_info(f"{channel_base_url}/videos", download=False)
        entries = info.get("entries", []) if isinstance(info, dict) else []

    for entry in entries:
        title = (entry.get("title") or "").strip()
        description = entry.get("description") or ""
        video_id = entry.get("id")
        if not title or not video_id:
            continue

        haystack = f"{title}\n{description}".lower()
        if not any(term in haystack for term in query_variants):
            continue

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        result = f"- **{title}**  \n  {video_url}"

        if include_snippets:
            snippets = fetch_transcript_snippets(
                video_id,
                expand_query(query)[:6],
                max_hits=3,
            )
            if snippets:
                result += "\n  " + "\n  ".join(f"• {snippet}" for snippet in snippets)

        matches.append(result)
        if len(matches) >= limit:
            break

    if not matches:
        return "No matching videos were found in the live fallback scan."
    return "LIVE_FALLBACK_RESULTS\n\n" + "\n".join(matches)


class SearchArgs(BaseModel):
    search_query: str = Field(..., description="Topic to search for in the selected channel")
    expand: bool = Field(default=True, description="Try related query variants when needed")
    max_variants: int = Field(default=5, ge=1, le=10)
    fallback_limit: int = Field(default=6, ge=1, le=12)
    fallback_snippets: bool = Field(default=True)


class FixedChannelYouTubeSearchTool(BaseTool):
    """CrewAI tool that searches one fixed YouTube channel with a live fallback."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "Search a YouTube Channel"
    description: str = (
        "Search within one fixed YouTube channel for topic-relevant videos. "
        "Uses semantic search first and a live metadata/transcript fallback when needed."
    )
    args_schema: Type[BaseModel] = SearchArgs

    channel_videos_url: str
    base_channel_url: str
    chosen_identifier: str

    _inner: Any = PrivateAttr(default=None)
    _run_accepts_url: bool = PrivateAttr(default=False)
    _run_accepts_handle: bool = PrivateAttr(default=False)

    def __init__(self, channel_videos_url: str):
        channel_videos_url = require_videos_url(channel_videos_url)
        base_channel_url = strip_videos_tab(channel_videos_url)

        handle_match = HANDLE_EXTRACT_RE.search(base_channel_url)
        ucid_match = UCID_EXTRACT_RE.search(base_channel_url)
        chosen_identifier = (
            f"@{handle_match.group(1)}"
            if handle_match
            else (ucid_match.group(1) if ucid_match else "")
        )
        if not chosen_identifier:
            raise ValueError(
                "Could not identify the channel. Use a /@handle/videos or /channel/UC.../videos URL."
            )

        super().__init__(
            channel_videos_url=channel_videos_url,
            base_channel_url=base_channel_url,
            chosen_identifier=chosen_identifier,
        )

        try:
            self._inner = YoutubeChannelSearchTool(youtube_channel_url=self.base_channel_url)
        except TypeError:
            try:
                self._inner = YoutubeChannelSearchTool(
                    youtube_channel_handle=self.chosen_identifier
                )
            except TypeError:
                self._inner = YoutubeChannelSearchTool(self.base_channel_url)

        try:
            signature = inspect.signature(self._inner.run)
            self._run_accepts_url = "youtube_channel_url" in signature.parameters
            self._run_accepts_handle = "youtube_channel_handle" in signature.parameters
        except Exception:
            self._run_accepts_handle = True

    def _call_inner(self, search_query: str) -> Any:
        if self._run_accepts_url:
            return self._inner.run(
                search_query=search_query,
                youtube_channel_url=self.base_channel_url,
            )
        if self._run_accepts_handle:
            return self._inner.run(
                search_query=search_query,
                youtube_channel_handle=self.chosen_identifier,
            )
        try:
            return self._inner.run(search_query=search_query)
        except TypeError:
            return self._inner.run(
                search_query=search_query,
                youtube_channel=self.base_channel_url,
            )

    def _run(
        self,
        search_query: str,
        expand: bool = True,
        max_variants: int = 5,
        fallback_limit: int = 6,
        fallback_snippets: bool = True,
        **_: Any,
    ) -> Any:
        result = self._call_inner(search_query)
        if not looks_empty(result):
            return result

        if expand:
            for variant in expand_query(search_query)[:max_variants]:
                if variant.lower() == search_query.lower():
                    continue
                expanded_result = self._call_inner(variant)
                if not looks_empty(expanded_result):
                    if isinstance(expanded_result, str):
                        return f"Matched related query: {variant}\n\n{expanded_result}"
                    return expanded_result

        return live_channel_search(
            self.base_channel_url,
            search_query,
            limit=fallback_limit,
            include_snippets=fallback_snippets,
        )


def build_agents(search_tool: FixedChannelYouTubeSearchTool, llm: LLM, memory: bool):
    researcher = Agent(
        role="YouTube Research Analyst",
        goal=(
            "Find the most relevant videos for {topic} in the selected channel and "
            "produce accurate, source-linked research notes."
        ),
        backstory=(
            "You specialize in extracting useful technical insights from video content "
            "without inventing facts or sources."
        ),
        tools=[search_tool],
        verbose=True,
        memory=memory,
        allow_delegation=False,
        llm=llm,
    )

    writer = Agent(
        role="Technical Content Writer",
        goal=(
            "Turn the research brief into a clear, accurate Markdown article about {topic}, "
            "grounded in the selected YouTube sources."
        ),
        backstory=(
            "You write concise technical content for professional audiences and clearly "
            "separate sourced information from general explanation."
        ),
        verbose=True,
        memory=memory,
        allow_delegation=False,
        llm=llm,
    )
    return researcher, writer


def build_tasks(
    researcher: Agent,
    writer: Agent,
    search_tool: FixedChannelYouTubeSearchTool,
    top_n: int,
    expand_queries: bool,
    max_variants: int,
    fallback_limit: int,
    fallback_snippets: bool,
):
    research_task = Task(
        description=(
            "Search the selected YouTube channel for videos relevant to '{topic}'. "
            f"Consider up to {top_n} strong matches. Extract the main points, URLs, and "
            "timestamped transcript snippets when available. Do not invent quotes or URLs.\n\n"
            f"Tool options: expand={expand_queries}, max_variants={max_variants}, "
            f"fallback_limit={fallback_limit}, fallback_snippets={fallback_snippets}."
        ),
        expected_output=(
            "A concise Markdown research brief containing the selected video title(s), URLs, "
            "why they are relevant, key insights, and transcript timestamps when available."
        ),
        tools=[search_tool],
        agent=researcher,
        output_file="research-brief.md",
    )

    writing_task = Task(
        description=(
            "Using only the research brief as the primary source, write a concise Markdown "
            "article on '{topic}'. Include an introduction, 3-5 useful sections, a short "
            "conclusion, and a Sources section listing the referenced YouTube video URLs. "
            "Do not fabricate citations, quotes, or claims."
        ),
        expected_output="A polished Markdown article grounded in the research brief.",
        context=[research_task],
        agent=writer,
        output_file="generated-blog-post.md",
    )

    return research_task, writing_task


def run_crew(
    channel_videos_url: str,
    topic: str,
    model_name: str,
    preset_key: str,
    search_depth: int,
    enable_memory: bool,
    expand_queries: bool,
    max_variants: int,
    fallback_limit: int,
    fallback_snippets: bool,
):
    search_tool = FixedChannelYouTubeSearchTool(channel_videos_url)
    llm = make_llm(model_name, preset_key)
    researcher, writer = build_agents(search_tool, llm, enable_memory)
    research_task, writing_task = build_tasks(
        researcher,
        writer,
        search_tool,
        search_depth,
        expand_queries,
        max_variants,
        fallback_limit,
        fallback_snippets,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        memory=enable_memory,
        cache=True,
        max_rpm=60,
        share_crew=False,
    )

    with st.spinner("Researching videos and drafting the article..."):
        return crew.kickoff(inputs={"topic": topic})


def read_output(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


st.title("🎬 Agentic YouTube Research Writer")
st.caption(
    "A CrewAI workflow that researches a selected YouTube channel and turns sourced findings into a Markdown article."
)

if not os.getenv("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY is not configured. Add it to a local .env file before running the agents.")

with st.sidebar:
    st.header("Configuration")
    channel_videos_url = st.text_input(
        "YouTube channel /videos URL",
        value=DEFAULT_CHANNEL_VIDEOS_URL,
        help="Example: https://www.youtube.com/@codebasics/videos",
    )
    model_name = st.selectbox(
        "OpenAI model",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0,
    )
    preset_key = st.radio(
        "Generation detail",
        options=list(TOKEN_PRESETS.keys()),
        index=1,
    )
    search_depth = st.slider("Videos to consider", 1, 10, 3)
    enable_memory = st.toggle("Enable CrewAI memory", value=False)
    expand_queries = st.toggle("Expand search terms when needed", value=True)
    max_variants = st.slider("Maximum query variants", 1, 10, 5)
    fallback_limit = st.slider("Live fallback results", 1, 12, 6)
    fallback_snippets = st.toggle("Include transcript snippets", value=True)


topic = st.text_input(
    "Topic",
    value="Retrieval-Augmented Generation",
    help="The agents will research this topic within the selected YouTube channel.",
)

if st.button("Run research workflow", type="primary"):
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is missing. Add it to .env and restart the app.")
        validated_url = require_videos_url(channel_videos_url)
        final_output = run_crew(
            validated_url,
            topic.strip(),
            model_name,
            preset_key,
            search_depth,
            enable_memory,
            expand_queries,
            max_variants,
            fallback_limit,
            fallback_snippets,
        )

        research_text = read_output("research-brief.md")
        article_text = read_output("generated-blog-post.md")

        st.success("Workflow completed.")
        research_tab, article_tab, raw_tab = st.tabs(
            ["Research Brief", "Generated Article", "Raw Crew Result"]
        )

        with research_tab:
            if research_text:
                st.markdown(research_text)
                st.download_button(
                    "Download research brief",
                    research_text,
                    "research-brief.md",
                    "text/markdown",
                )
            else:
                st.info("No research brief was generated.")

        with article_tab:
            if article_text:
                st.markdown(article_text)
                st.download_button(
                    "Download article",
                    article_text,
                    "generated-blog-post.md",
                    "text/markdown",
                )
            else:
                st.info("No article was generated.")

        with raw_tab:
            st.code(str(final_output))

    except Exception as exc:
        st.error(str(exc))
        with st.expander("Technical details"):
            st.exception(exc)

with st.expander("How it works"):
    st.markdown(
        """
1. The research agent searches the selected YouTube channel for the requested topic.
2. If semantic search returns nothing useful, the app scans recent channel uploads with `yt-dlp`.
3. When available, transcript snippets and timestamps are added to the research brief.
4. A writing agent turns the sourced research into a Markdown article.

The generated content should still be reviewed before publication.
        """
    )
