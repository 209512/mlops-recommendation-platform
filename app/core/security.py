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
        self.secret_key = settings.secret_key or "default-secret-key"  # None 처리
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30

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

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)

        to_encode.update({"exp": expire})

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
            Optional[Dict[str, Any]]: 토큰 페이로드 또는 None
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.PyJWTError:
            logger.warning("Invalid token")
            return None

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
security = SecurityManager()
