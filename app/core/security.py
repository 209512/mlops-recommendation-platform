import logging
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import jwt
from cryptography.fernet import Fernet
from passlib.context import CryptContext

from app.core.config import settings

logger = logging.getLogger(__name__)

# 비밀번호 해싱 컨텍스트
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,
)


class Environment(str, Enum):
    """환경 타입 열거형"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class SecurityConfig:
    """환경별 보안 설정 클래스"""

    def __init__(self, env: Environment):
        self.env = env
        self._setup_config()

    def _setup_config(self) -> None:
        """환경별 설정 초기화"""
        if self.env == Environment.PRODUCTION:
            self.access_token_expire_minutes = 15
            self.refresh_token_expire_days = 7
            self.require_https = True
            self.session_timeout = 30  # minutes
            self.max_login_attempts = 5
            self.lockout_duration = 15  # minutes
        elif self.env == Environment.STAGING:
            self.access_token_expire_minutes = 30
            self.refresh_token_expire_days = 14
            self.require_https = True
            self.session_timeout = 60
            self.max_login_attempts = 10
            self.lockout_duration = 10
        elif self.env == Environment.TESTING:
            self.access_token_expire_minutes = 60
            self.refresh_token_expire_days = 1
            self.require_https = False
            self.session_timeout = 120
            self.max_login_attempts = 100
            self.lockout_duration = 1
        else:  # DEVELOPMENT
            self.access_token_expire_minutes = 60
            self.refresh_token_expire_days = 30
            self.require_https = False
            self.session_timeout = 120
            self.max_login_attempts = 50
            self.lockout_duration = 5


class SecurityManager:
    """보안 관리자 - 환경별 설정 분리"""

    def __init__(self, env: Environment | None = None) -> None:
        # 환경 자동 감지
        self.env = env or self._detect_environment()
        self.config = SecurityConfig(self.env)

        # 시크릿 키 관리
        self.secret_key = self._get_secret_key()
        self.encryption_key = self._get_encryption_key()

        # JWT 설정
        self.algorithm = "HS256"
        self.issuer = "mlops-recommendation-platform"

        # 암호화 관리자
        self.cipher_suite = Fernet(self.encryption_key)

        # 로그인 시도 추적
        self._login_attempts: dict[str, dict[str, Any]] = {}

        self._validate_security_requirements()

    def _detect_environment(self) -> Environment:
        """환경 자동 감지"""
        env_var = os.getenv("ENVIRONMENT", "").lower()

        if env_var == "production":
            return Environment.PRODUCTION
        elif env_var == "staging":
            return Environment.STAGING
        elif env_var == "test":
            return Environment.TESTING
        else:
            return Environment.DEVELOPMENT

    def _get_secret_key(self) -> str:
        """환경별 시크릿 키 획득"""
        if self.env == Environment.PRODUCTION:
            if not settings.secret_key or settings.secret_key in [
                "default-secret-key",
                "your-secret-key-here-change-in-production",
                "dev-insecure-key",
            ]:
                raise ValueError(
                    "Production requires a secure SECRET_KEY. "
                    "Set a strong, unique secret key in environment variables."
                )
            return settings.secret_key
        elif self.env == Environment.TESTING:
            return "test-secret-key-for-testing-only"
        else:
            # 개발 환경에서는 경고와 함께 기본값 사용
            if not settings.secret_key or settings.secret_key == "default-secret-key":
                logger.warning(
                    f"Using development secret key in {self.env.value} environment. "
                    "This is insecure and should only be used for development."
                )
                return "dev-secret-key-change-in-production"
            return settings.secret_key

    def _get_encryption_key(self) -> bytes:
        """암호화 키 획득 또는 생성"""
        key_env = f"ENCRYPTION_KEY_{self.env.value.upper()}"
        key = os.getenv(key_env)

        if key:
            return key.encode()

        # 개발/테스트 환경에서는 키 생성
        if self.env in [Environment.DEVELOPMENT, Environment.TESTING]:
            generated_key = Fernet.generate_key()
            logger.warning(
                f"Generated encryption key for {self.env.value}. "
                f"Set {key_env} environment variable for persistence."
            )
            return generated_key

        raise ValueError(
            f"Encryption key required for {self.env.value} environment. "
            f"Set {key_env} environment variable."
        )

    def _validate_security_requirements(self) -> None:
        """환경별 보안 요구사항 검증"""
        if self.env == Environment.PRODUCTION:
            if not self.config.require_https:
                logger.error("HTTPS must be required in production")
            if self.config.access_token_expire_minutes > 30:
                logger.warning("Consider shorter token expiration in production")

    def is_account_locked(self, identifier: str) -> bool:
        """계정 잠금 상태 확인"""
        if identifier not in self._login_attempts:
            return False

        attempts = self._login_attempts[identifier]
        if attempts["count"] >= self.config.max_login_attempts:
            lock_time = attempts.get("locked_until")
            if lock_time and datetime.utcnow() < lock_time:
                return True
            else:
                # 잠금 시간 지남 - 카운터 리셋
                del self._login_attempts[identifier]
                return False
        return False

    def record_login_attempt(self, identifier: str, success: bool = False) -> None:
        """로그인 시도 기록"""
        if identifier not in self._login_attempts:
            self._login_attempts[identifier] = {"count": 0}

        if success:
            # 성공 시 카운터 리셋
            del self._login_attempts[identifier]
        else:
            self._login_attempts[identifier]["count"] += 1

            # 최대 시도 횟수 초과 시 잠금
            if self._login_attempts[identifier]["count"] >= self.config.max_login_attempts:
                lock_until = datetime.utcnow() + timedelta(minutes=self.config.lockout_duration)
                self._login_attempts[identifier]["locked_until"] = lock_until
                logger.warning(f"Account {identifier} locked due to multiple failed attempts")

    def create_access_token(
        self, data: dict[str, Any], expires_delta: timedelta | None = None
    ) -> str:
        """JWT 액세스 토큰 생성"""
        to_encode = data.copy()
        now = datetime.utcnow()

        if expires_delta:
            expire = now + expires_delta
        else:
            expire = now + timedelta(minutes=self.config.access_token_expire_minutes)

        to_encode.update(
            {
                "exp": expire,
                "iat": now,
                "iss": self.issuer,
                "type": "access",
                "env": self.env.value,
                "ver": "1.0",  # 토큰 버전
            }
        )

        token = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm,
        )

        # PyJWT 2.0+는 이미 문자열 반환
        return token if isinstance(token, str) else token.decode("utf-8")

    def verify_token(self, token: str) -> dict[str, Any] | None:
        """JWT 토큰 검증"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                issuer=self.issuer,
                options={"require": ["exp", "iat", "iss", "type"]},
            )

            # 토큰 타입 및 환경 검증
            if payload.get("type") != "access":
                logger.warning("Invalid token type")
                return None

            # 환경 불일치 검증 (프로덕션에서만)
            if self.env == Environment.PRODUCTION and payload.get("env") != self.env.value:
                logger.warning("Token environment mismatch")
                return None

            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidIssuerError:
            logger.warning("Invalid token issuer")
            return None
        except jwt.MissingRequiredClaimError as e:
            logger.warning(f"Missing required claim: {e}")
            return None
        except jwt.PyJWTError as e:
            logger.warning(f"Invalid token: {e}")
            return None

    def create_refresh_token(
        self, data: dict[str, Any], expires_delta: timedelta | None = None
    ) -> str:
        """JWT 리프레시 토큰 생성"""
        to_encode = data.copy()
        now = datetime.utcnow()

        if expires_delta:
            expire = now + expires_delta
        else:
            expire = now + timedelta(days=self.config.refresh_token_expire_days)

        to_encode.update(
            {
                "exp": expire,
                "iat": now,
                "iss": self.issuer,
                "type": "refresh",
                "env": self.env.value,
            }
        )

        token = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm,
        )

        return token if isinstance(token, str) else token.decode("utf-8")

    def hash_password(self, password: str) -> str:
        """비밀번호 해싱"""
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """비밀번호 검증"""
        return pwd_context.verify(plain_password, hashed_password)

    def encrypt_sensitive_data(self, data: str) -> str:
        """민감 데이터 암호화"""
        return self.cipher_suite.encrypt(data.encode()).decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """민감 데이터 복호화"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()

    def get_security_headers(self) -> dict[str, str]:
        """보안 헤더 반환"""
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }

        if self.config.require_https:
            headers.update(
                {
                    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                    "Content-Security-Policy": "default-src 'self'",
                }
            )

        return headers


def create_security_manager(env: Environment | None = None) -> SecurityManager:
    """환경별 보안 관리자 생성"""
    return SecurityManager(env)


# 전역 보안 관리자 인스턴스
try:
    security = create_security_manager()
    logger.info(f"Security manager initialized for {security.env.value} environment")
except ValueError as e:
    logger.error(f"Failed to initialize SecurityManager: {e}")

    # 개발 환경에서만 fallback 허용
    env = os.getenv("ENVIRONMENT", "development").lower()
    if env in ["development", "dev"]:
        logger.warning("Creating fallback SecurityManager for development")
        security = create_security_manager(Environment.DEVELOPMENT)
    else:
        logger.critical("Cannot initialize security in production environment")
        raise


def get_security_manager() -> SecurityManager:
    """보안 관리자 의존성 주입"""
    return security
