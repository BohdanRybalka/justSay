from app.core.types import ProviderMode
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    mode: ProviderMode = ProviderMode.CLOUD

    groq_api_key: str = ""
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:1.7b"

    model_config = SettingsConfigDict(env_prefix="JUSTSAY_LLM_", env_file=".env", extra="ignore")
