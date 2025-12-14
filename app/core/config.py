from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """현대적 설정 관리 - 타입 안전성과 환경 변수 자동 바인딩"""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",  # 추가 환경 변수 무시
    )

    # 환경 타입
    environment: Literal["local", "test", "staging", "production"] = "local"

    # Application
    app_name: str = "MLOps Recommendation Platform"
    debug: bool = False
    version: str = "0.1.0"
    secret_key: str = Field(...)
    log_level: str = "INFO"

    # Database
    database_url: str = Field(...)
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=5432)
    db_name: str = Field(default="mlops")
    db_user: str = Field(default="postgres")
    postgres_password: str = Field(...)

    # Redis
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_url: str = Field(default="redis://localhost:6379/0")

    # ALS Model Configuration
    als_model_version: str = Field(default="v1")
    model_storage_path: str = Field(default="./models")

    # MLflow
    mlflow_tracking_uri: str = Field(...)
    mlflow_experiment_name: str = Field(default="als_recommendation")
    mlflow_backend_store_uri: str = Field(...)
    mlflow_default_artifact_root: str = Field(default="./mlruns")

    # FastAPI
    allowed_hosts: list[str] = Field(default=["localhost", "127.0.0.1"])

    # CORS 설정
    cors_origins: list[str] = Field(default=[])
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: list[str] = Field(default=["GET", "POST", "PUT", "DELETE"])
    cors_allow_headers: list[str] = Field(default=["Content-Type", "Authorization"])

    # Celery
    celery_broker_url: str = Field(...)
    celery_result_backend: str = Field(...)

    # Monitoring
    prometheus_port: int = Field(default=8001)
    grafana_port: int = Field(default=3000)

    # AWS
    aws_access_key_id: str = Field(...)
    aws_secret_access_key: str = Field(...)
    aws_region: str = Field(default="ap-northeast-2")
    ecr_registry: str = Field(...)

    # SonarQube
    sonar_host_url: str = Field(...)
    sonar_token: str = Field(...)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_test(self) -> bool:
        return self.environment == "test"


settings = Settings()
