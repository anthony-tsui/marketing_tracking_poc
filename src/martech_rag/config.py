"""Application settings loaded from environment / .env."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")
    notion_token: str = Field(..., alias="NOTION_TOKEN")

    openrouter_base_url: str = Field(
        "https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    chat_model: str = Field(
        "deepseek/deepseek-v4-flash-0731",
        alias="CHAT_MODEL",
    )
    retrieval_model: str = Field("z-ai/glm-5.2", alias="RETRIEVAL_MODEL")
    embed_model: str = Field(
        "nvidia/nemotron-3-embed-1b:free",
        alias="EMBED_MODEL",
    )

    # Notion IDs (My Second Brain)
    notion_notes_database_id: str = Field(
        "09e702c5-d853-8239-8eac-018c27489af5",
        alias="NOTION_NOTES_DATABASE_ID",
    )
    # Notes data source (required by newer Notion API query endpoints)
    notion_notes_data_source_id: str = Field(
        "228702c5-d853-83b9-b8d0-87dc93a7c17f",
        alias="NOTION_NOTES_DATA_SOURCE_ID",
    )
    notion_project_id: str = Field(
        "3ba702c5-d853-80f2-ab52-cdfeb5d46762",
        alias="NOTION_PROJECT_ID",
    )
    notion_martech_tag_id: str = Field(
        "367702c5-d853-803d-b386-f2e57b13b4af",
        alias="NOTION_MARTECH_TAG_ID",
    )

    chroma_dir: Path = Field(default=ROOT_DIR / "data" / "chroma", alias="CHROMA_DIR")
    crawl_state_path: Path = Field(
        default=ROOT_DIR / "data" / "crawl_state.json",
        alias="CRAWL_STATE_PATH",
    )
    eval_testset_path: Path = Field(
        default=ROOT_DIR / "data" / "eval" / "testset.json",
        alias="EVAL_TESTSET_PATH",
    )

    retrieval_top_k: int = Field(8, alias="RETRIEVAL_TOP_K")
    rerank_top_n: int = Field(5, alias="RERANK_TOP_N")
    expand_n: int = Field(3, alias="EXPAND_N")
    chunk_size: int = Field(500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(80, alias="CHUNK_OVERLAP")
    parent_chunk_size: int = Field(2800, alias="PARENT_CHUNK_SIZE")
    crawl_max_concurrency: int = Field(3, alias="CRAWL_MAX_CONCURRENCY")
    embed_batch_size: int = Field(16, alias="EMBED_BATCH_SIZE")
    # Default off: Google docs work via HTTP; avoids Playwright install failures
    crawl_use_browser: bool = Field(False, alias="CRAWL_USE_BROWSER")
    # Notion markdown/query calls often exceed the SDK's 60s default
    notion_timeout_ms: int = Field(180_000, alias="NOTION_TIMEOUT_MS")
    notion_max_retries: int = Field(5, alias="NOTION_MAX_RETRIES")


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
