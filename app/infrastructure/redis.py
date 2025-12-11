import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

import redis
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from app.core.config import settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # mypy에서만 제네릭 타입 사용
    BaseRedis = Redis[bytes]
else:
    # 런타임에서는 일반 타입 사용
    BaseRedis = Redis


class RedisClient:
    """싱글톤 Redis 클라이언트"""

    _instance: Optional["RedisClient"] = None
    _client: BaseRedis | None = None
    _last_health_check: float = 0
    _health_check_interval: int = 30  # 30초마다 헬스 체크
    _lock = threading.Lock()  # 스레드 세이프를 위한 클래스 락

    def __new__(cls) -> "RedisClient":
        # Double-checked locking 패턴으로 스레드 세이프한 싱글톤 구현
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._client = redis.Redis(
                        host=settings.redis_host,
                        port=settings.redis_port,
                        decode_responses=False,
                        socket_connect_timeout=5,
                        socket_timeout=5,
                        retry_on_timeout=True,  # 타임아웃 시 재시도 활성화
                        health_check_interval=30,  # 커넥션 풀링 헬스 체크
                    )
        return cls._instance

    def get_client(self) -> BaseRedis:
        """Redis 클라이언트 인스턴스 반환"""
        if self._client is None:
            raise RuntimeError("Redis client not initialized")
        return self._client

    def is_connection_healthy(self, max_retries: int = 3) -> bool:
        """Redis 연결 상태 확인 (재시도 로직 포함)"""
        current_time = time.time()

        # 캐시된 결과 사용 (30초 이내)
        if current_time - self._last_health_check < self._health_check_interval:
            return True

        for attempt in range(max_retries):
            try:
                result = self.get_client().ping()
                self._last_health_check = current_time
                return result
            except (RedisConnectionError, RedisTimeoutError) as e:
                if attempt == max_retries - 1:
                    logger.error(f"Redis health check failed after {max_retries} attempts: {e}")
                    return False
                logger.warning(f"Redis health check attempt {attempt + 1} failed, retrying...")
                time.sleep(0.5 * (attempt + 1))  # 점진적 대기

        return False

    def get_with_retry(self, key: str, max_retries: int = 3) -> bytes | None:
        """재시도 로직 포함 GET 연산"""
        for attempt in range(max_retries):
            try:
                return self.get_client().get(key)
            except (RedisConnectionError, RedisTimeoutError) as e:
                if attempt == max_retries - 1:
                    logger.error(
                        f"Redis GET failed for key {key} after {max_retries} attempts: {e}"
                    )
                    return None
                logger.warning(f"Redis GET attempt {attempt + 1} failed for key {key}, retrying...")
                time.sleep(0.1 * (attempt + 1))

        return None

    def set_with_retry(self, key: str, value: bytes, max_retries: int = 3) -> bool:
        """재시도 로직 포함 SET 연산"""
        for attempt in range(max_retries):
            try:
                result = self.get_client().set(key, value)
                if result is None:
                    logger.warning(f"Redis SET returned None for key {key}")
                    return False
                return result
            except (RedisConnectionError, RedisTimeoutError) as e:
                if attempt == max_retries - 1:
                    logger.error(
                        f"Redis SET failed for key {key} after {max_retries} attempts: {e}"
                    )
                    return False
                logger.warning(f"Redis SET attempt {attempt + 1} failed for key {key}, retrying...")
                time.sleep(0.1 * (attempt + 1))

        return False


redis_client = RedisClient()


def get_redis_client() -> BaseRedis:
    """Redis 클라이언트 의존성 주입"""
    return redis_client.get_client()


def is_redis_healthy() -> bool:
    """Redis 헬스 체크 유틸리티 함수"""
    return redis_client.is_connection_healthy()
