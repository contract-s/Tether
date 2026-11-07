from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    supervisor_model: str = "claude-opus-5-5"
    specialist_model: str = "claude-sonnet-5-5"
    reviewer_model: str = "claude-sonnet-5-5"

    database_url: str = "postgresql+psycopg://agentorch:agentorch@localhost:5432/agentorch"
    redis_url: str = "redis://localhost:6379/0"
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # Tools
    workspace_dir: str = "data/workspace"
    sandbox_image: str = "python:3.11-slim"
    sandbox_timeout_seconds: int = 30
    # Model spend allowed per specialist attempt before its tools are switched off.
    specialist_budget_usd: float = 0.40

    # Where task runs happen: "thread" (inside the API process) or "celery" (workers).
    task_runner: str = "thread"
    task_visibility_timeout_seconds: int = 3600

    # Observability: spans also go to an OTLP collector (e.g. Jaeger) when this is set.
    otel_exporter_otlp_endpoint: str = ""
    otel_service_name: str = "agentorch"

    # Human review: optional Slack-style webhook for "review needed" notifications.
    review_webhook_url: str = ""

    plan_confidence_threshold: float = 0.7
    review_quality_threshold: float = 0.7
    max_specialist_retries: int = 2


settings = Settings()
