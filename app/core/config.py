import os
import secrets
from pathlib import Path
from typing import Any, Literal

from cryptography.fernet import Fernet
from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """현대적 설정 관리"""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # 환경 타입
    environment: Literal["local", "test", "staging", "production"] = Field(default="local")

    # Application
    app_name: str = Field(default="MLOps Recommendation Platform")
    debug: bool = Field(default=False)
    version: str = Field(default="0.1.0")
    secret_key: str = Field(...)
    log_level: str = Field(default="INFO")

    # 보안 상수
    min_secret_key_length: int = Field(default=32, description="최소 시크릿 키 길이")
    default_session_timeout: int = Field(default=30, description="기본 세션 타임아웃 (분)")
    default_login_attempts: int = Field(default=5, description="기본 최대 로그인 시도 횟수")
    default_lockout_duration: int = Field(default=15, description="기본 계정 잠금 지속 시간 (분)")

    # 기본 시크릿 키 목록
    default_secret_keys: list[str] = Field(
        default=[
            "default-secret-key",
            "your-secret-key-here-change-in-production",
            "dev-insecure-key",
        ],
        description="사용 금지 기본 시크릿 키 목록",
    )

    # 기본 암호화 키
    encryption_key: str | None = Field(
        default=None, description="Fernet 암호화 키 (자동 생성 가능)"
    )

    # 키 파일 경로
    encryption_key_file: Path | None = Field(default=None, description="암호화 키 저장 파일 경로")

    # 키 로테이션 설정
    key_rotation_enabled: bool = Field(default=False, description="암호화 키 자동 로테이션 활성화")
    key_rotation_days: int = Field(default=90, description="키 로테이션 주기 (일)")

    # 키 백업 설정
    key_backup_enabled: bool = Field(default=False, description="암호화 키 백업 활성화")
    key_backup_path: Path | None = Field(default=None, description="키 백업 저장 경로")

    # 보안 설정
    require_https: bool = Field(default=False, description="HTTPS 강제 요구")
    session_timeout_minutes: int = Field(default=30, description="세션 타임아웃 (분)")
    max_login_attempts: int = Field(default=5, description="최대 로그인 시도 횟수")
    lockout_duration_minutes: int = Field(default=15, description="계정 잠금 지속 시간 (분)")

    # 환경별 보안 설정
    production_session_timeout: int = Field(default=15, description="프로덕션 세션 타임아웃 (분)")
    production_max_login_attempts: int = Field(
        default=3, description="프로덕션 최대 로그인 시도 횟수"
    )
    production_lockout_duration: int = Field(
        default=30, description="프로덕션 계정 잠금 지속 시간 (분)"
    )

    test_session_timeout: int = Field(default=120, description="테스트 세션 타임아웃 (분)")
    test_max_login_attempts: int = Field(default=100, description="테스트 최대 로그인 시도 횟수")
    test_lockout_duration: int = Field(default=1, description="테스트 계정 잠금 지속 시간 (분)")

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

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str, info: ValidationInfo) -> str:
        """시크릿 키 검증"""
        if not v:
            raise ValueError("SECRET_KEY는 비어있을 수 없습니다")

        if len(v) < info.data.get("min_secret_key_length", 32):
            raise ValueError(
                f"SECRET_KEY는 최소 "
                f"{info.data.get('min_secret_key_length', 32)}자 이상이어야 합니다"
            )

        # 기본 키 사용 방지
        default_keys = info.data.get("default_secret_keys", [])
        if v in default_keys:
            if info.data.get("environment") == "production":
                raise ValueError("프로덕션 환경에서는 기본 SECRET_KEY를 사용할 수 없습니다")

        return v

    @field_validator("encryption_key")
    @classmethod
    def validate_encryption_key(cls, v: str | None) -> str | None:
        """암호화 키 검증"""
        if v is not None:
            try:
                Fernet(v.encode())
            except Exception:
                raise ValueError("유효하지 않은 Fernet 암호화 키입니다") from None
        return v

    def get_encryption_key(self) -> str:
        """암호화 키 가져오기 또는 생성"""
        if self.encryption_key:
            return self.encryption_key

        # 파일에서 키 읽음
        if self.encryption_key_file and self.encryption_key_file.exists():
            with open(self.encryption_key_file, "rb") as f:
                key = f.read().decode()
            return key

        # 새 키 생성
        key = Fernet.generate_key().decode()

        # 파일에 저장
        if self.encryption_key_file:
            self.encryption_key_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.encryption_key_file, "wb") as f:
                f.write(key.encode())
            os.chmod(self.encryption_key_file, 0o600)

        return key

    def rotate_encryption_key(self) -> str:
        """암호화 키 로테이션"""
        if not self.key_rotation_enabled:
            raise ValueError("키 로테이션이 비활성화되어 있습니다.")

        # 기존 키 백업
        if self.key_backup_enabled and self.key_backup_path:
            old_key = self.get_encryption_key()
            backup_file = self.key_backup_path / f"encryption_key_{secrets.token_hex(8)}.key"
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            with open(backup_file, "wb") as f:
                f.write(old_key.encode())
            os.chmod(backup_file, 0o600)

        # 새 키 생성
        new_key = Fernet.generate_key().decode()

        # 파일에 저장
        if self.encryption_key_file:
            with open(self.encryption_key_file, "wb") as f:
                f.write(new_key.encode())

        return new_key

    def get_security_config(self) -> dict[str, Any]:
        """환경별 보안 설정 반환"""
        base_config = {
            "require_https": self.require_https,
            "session_timeout": self.session_timeout_minutes,
            "max_login_attempts": self.max_login_attempts,
            "lockout_duration": self.lockout_duration_minutes,
        }

        # 환경별 설정 조정
        if self.is_production:
            base_config.update(
                {
                    "require_https": True,
                    "session_timeout": self.production_session_timeout,
                    "max_login_attempts": self.production_max_login_attempts,
                    "lockout_duration": self.production_lockout_duration,
                }
            )
        elif self.is_test:
            base_config.update(
                {
                    "require_https": False,
                    "session_timeout": self.test_session_timeout,
                    "max_login_attempts": self.test_max_login_attempts,
                    "lockout_duration": self.test_lockout_duration,
                }
            )

        return base_config

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_test(self) -> bool:
        return self.environment == "test"


# 전역 설정 인스턴스
settings = Settings()
