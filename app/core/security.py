import logging
from datetime import datetime, timedelta
from typing import Any

import jwt
from passlib.context import CryptContext

from app.core.config import settings

logger = logging.getLogger(__name__)

# 비밀번호 해싱 컨텍스트
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SecurityManager:
    """보안 관리자 - JWT 토큰 및 비밀번호 관리"""

    def __init__(self) -> None:
        # 기본 secret_key 사용 방지 - 프로덕션 환경에서는 강제 검증
        if not settings.secret_key:
            raise ValueError(
                "SECRET_KEY must be set in environment variables. "
                "Cannot use empty secret key for security reasons."
            )

        if settings.secret_key == "default-secret-key":
            if settings.is_production:
                raise ValueError(
                    "Default secret key cannot be used in production. "
                    "Please set a secure SECRET_KEY environment variable."
                )
            else:
                logger.warning(
                    r"Using default secret key.\ "
                    "This is insecure and should only be used in development."
                )

        self.secret_key = settings.secret_key
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.issuer = "mlops-recommendation-platform"  # 토큰 발행자 식별

    def create_access_token(
        self, data: dict[str, Any], expires_delta: timedelta | None = None
    ) -> str:
        """
        JWT 액세스 토큰 생성

        Args:
            data: 토큰에 포함할 데이터
            expires_delta: 만료 시간 델타

        Returns:
            str: JWT 토큰
        """
        to_encode = data.copy()
        now = datetime.utcnow()

        if expires_delta:
            expire = now + expires_delta
        else:
            expire = now + timedelta(minutes=self.access_token_expire_minutes)

        # JWT 클레임 개선 - 표준 클레임 추가
        to_encode.update(
            {
                "exp": expire,
                "iat": now,  # 발행 시간 추가
                "iss": self.issuer,  # 발행자 정보
                "type": "access",  # 토큰 타입 명시
            }
        )

        token_bytes: bytes = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm,
        )
        token: str = token_bytes.decode("utf-8")
        return token

    def verify_token(self, token: str) -> dict[str, Any] | None:
        """
        JWT 토큰 검증

        Args:
            token: JWT 토큰

        Returns:
            Optional[Dict[str, Any]]: 토큰 페이로드 또은 None
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                issuer=self.issuer,  # 발행자 검증
                options={"require": ["exp", "iat", "iss"]},  # 필수 클레임 검증
            )

            # 토큰 타입 검증
            if payload.get("type") != "access":
                logger.warning("Invalid token type")
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
        except jwt.PyJWTError:
            logger.warning("Invalid token")
            return None

    def create_refresh_token(
        self, data: dict[str, Any], expires_delta: timedelta | None = None
    ) -> str:
        """
        JWT 리프레시 토큰 생성 (선택적 기능)

        Args:
            data: 토큰에 포함할 데이터
            expires_delta: 만료 시간 델타

        Returns:
            str: JWT 리프레시 토큰
        """
        to_encode = data.copy()
        now = datetime.utcnow()

        if expires_delta:
            expire = now + expires_delta
        else:
            # 리프레시 토큰은 더 긴 유효기간
            expire = now + timedelta(days=7)

        to_encode.update({"exp": expire, "iat": now, "iss": self.issuer, "type": "refresh"})

        token_bytes: bytes = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm,
        )
        return token_bytes.decode("utf-8")

    def hash_password(self, password: str) -> str:
        """
        비밀번호 해싱

        Args:
            password: 평문 비밀번호

        Returns:
            str: 해시된 비밀번호
        """
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        비밀번호 검증

        Args:
            plain_password: 평문 비밀번호
            hashed_password: 해시된 비밀번호

        Returns:
            bool: 검증 결과
        """
        return pwd_context.verify(plain_password, hashed_password)


# 전역 보안 관리자 인스턴스
try:
    security = SecurityManager()
except ValueError as e:
    logger.error(f"Failed to initialize SecurityManager: {e}")
    # 개발 환경에서는 기본 인스턴스 생성, 프로덕션에서는 앱 시작 실패
    if not settings.is_production:
        logger.warning("Creating insecure SecurityManager for development")
        security = SecurityManager()
        security.secret_key = "dev-insecure-key"
    else:
        raise
