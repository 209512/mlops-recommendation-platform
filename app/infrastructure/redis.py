import logging
from typing import Optional

import redis
from redis import Redis

from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """싱글톤 Redis 클라이언트"""

    _instance: Optional["RedisClient"] = None
    _client: Redis | None = None

    def __new__(cls) -> "RedisClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
        return cls._instance

    def get_client(self) -> Redis:
        """Redis 클라이언트 인스턴스 반환"""
        if self._client is None:
            raise RuntimeError("Redis client not initialized")
        return self._client

    def ping(self) -> bool:
        """Redis ping 테스트"""
        try:
            return self.get_client().ping()
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False

    def health_check(self) -> bool:
        """Redis 헬스 체크"""
        try:
            self.get_client().ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

        # 전역 Redis 클라이언트 인스턴스


redis_client = RedisClient()


def get_redis_client() -> Redis:
    """Redis 클라이언트 의존성 주입"""
    return redis_client.get_client()
