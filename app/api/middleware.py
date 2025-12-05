import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Prometheus 메트릭
request_counter = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"]
)

request_latency = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"]
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """요청 메트릭 수집 미들웨어"""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start_time = time.time()

        response = await call_next(request)

        # 메트릭 기록
        method = request.method
        endpoint = request.url.path
        status_code = str(response.status_code)

        request_counter.labels(method=method, endpoint=endpoint, status_code=status_code).inc()

        request_latency.labels(method=method, endpoint=endpoint).observe(time.time() - start_time)

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
