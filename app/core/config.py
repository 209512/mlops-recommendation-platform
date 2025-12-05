from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """현대적 설정 관리 - 타입 안전성과 환경 변수 자동 바인딩"""

    # Application
    app_name: str = "MLOps Recommendation Platform"
    debug: bool = False
    version: str = "0.1.0"
    secret_key: str | None = None
    log_level: str = "INFO"

    # Database
    database_url: str | None = None
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str | None = None
    db_user: str | None = None
    db_password: str | None = None

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_url: str | None = None

    # ALS Model Configuration
    als_model_version: str = "v1"
    model_storage_path: str = "./models"

    # MLflow
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "als_recommendation"

    # FastAPI
    allowed_hosts: list[str] = ["localhost", "127.0.0.1"]

    # Celery
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None

    # Monitoring
    prometheus_port: int = 8001
    grafana_port: int = 3000

    # AWS
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "ap-northeast-2"
    ecr_registry: str | None = None

    # SonarQube
    sonar_host_url: str | None = None
    sonar_token: str | None = None

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
