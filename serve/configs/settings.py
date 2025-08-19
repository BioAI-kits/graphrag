from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    server_port: int = 8000
    cors_allowed_origins: list = ["*"]  # Edit the list to restrict access.
    root: str = "../graphrag_zh"
    data: str = "../graphrag_zh/output"
    community_level: int = 2
    dynamic_community_selection: bool = False
    response_type: str = "Multiple Paragraphs"
    settings_yaml: str = "../graphrag_zh/settings.yaml"
    ENV_PATH: str = "../graphrag_zh/.env"

    @property
    def website_address(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"


settings = Settings()
