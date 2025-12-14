import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


def _group_endpoint(path: str) -> str:
    """
    엔드포인트 경로를 그룹화하여 메트릭 카디널리티 문제 해결

    Args:
        path: 원본 요청 경로

    Returns:
        그룹화된 경로
    """
    # 정적 파일 및 헬스 체크
    if path in ["/health", "/metrics", "/docs", "/openapi.json"]:
        return path

    # API 경로 그룹화
    if path.startswith("/api/v1/"):
        parts = path.split("/")
        if len(parts) >= 4:
            # /api/v1/recommendations/123 -> /api/v1/recommendations/
            # /api/v1/users/456/preferences -> /api/v1/users/
            return f"/api/v1/{parts[3]}/"

    # MLflow 경로 그룹화
    if path.startswith("/api/v1/mlflow/"):
        return "/api/v1/mlflow/"

    return path


# Prometheus 메트릭 - 그룹화된 엔드포인트 사용
request_counter = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint_group", "status_code"]
)

request_latency = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "endpoint_group"]
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """요청 메트릭 수집 미들웨어"""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start_time = time.time()

        response = await call_next(request)

        # 메트릭 기록 - 그룹화된 엔드포인트 사용
        method = request.method
        raw_endpoint = request.url.path
        endpoint_group = _group_endpoint(raw_endpoint)
        status_code = str(response.status_code)

        request_counter.labels(
            method=method, endpoint_group=endpoint_group, status_code=status_code
        ).inc()

        request_latency.labels(method=method, endpoint_group=endpoint_group).observe(
            time.time() - start_time
        )

        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """요청 로깅 미들웨어"""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start_time = time.time()

        logger.info(f"Request started: {request.method} {request.url.path}")

        response = await call_next(request)

        process_time = time.time() - start_time
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"- Status: {response.status_code} - Time: {process_time:.4f}s"
        )

        return response
