import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from app.api.middleware import LoggingMiddleware, MetricsMiddleware
from app.api.v1.monitoring import router as monitoring_router
from app.api.v1.recommendations import router as recommendations_router
from app.api.v1.training import router as training_router
from app.core.config import settings
from app.core.exception import MLOpsError
from app.infrastructure.database import async_engine
from app.infrastructure.redis import redis_client

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """애플리케이션 생명주기 관리"""
    # Startup 로직
    logger.info("Starting MLOps Recommendation Platform...")

    try:
        # 데이터베이스 연결 확인
        async with async_engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
            logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

    # Redis 연결 확인
    try:
        if redis_client:
            redis_client.ping()
            logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")

    # 모델 로딩 시도
    try:
        # TODO: 모델 로딩 로직 구현
        logger.info("Model loading skipped - not implemented")
    except Exception as e:
        logger.warning(f"Model loading failed: {e}")

    logger.info("MLOps Recommendation Platform started successfully")

    yield

    # Shutdown 로직
    logger.info("Shutting down MLOps Recommendation Platform...")

    try:
        await async_engine.dispose()
    except Exception as e:
        logger.warning(f"Engine disposal failed: {e}")

    try:
        if redis_client:
            await redis_client.aclose() if hasattr(redis_client, "aclose") else None
    except Exception as e:
        logger.warning(f"Redis close failed: {e}")

    logger.info("MLOps Recommendation Platform shut down complete")


# FastAPI 앱 생성
app = FastAPI(
    title="MLOps Recommendation Platform",
    description="FastAPI-based ALS recommendation system with MLOps",
    version="0.1.0",
    lifespan=lifespan,
)

# 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_hosts,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts,
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(MetricsMiddleware)


# 예외 핸들러
@app.exception_handler(MLOpsError)
async def mlops_exception_handler(request: Request, exc: MLOpsError) -> JSONResponse:
    """MLOps 전용 예외 핸들러"""
    return JSONResponse(
        status_code=getattr(exc, "status_code", 500),
        content={
            "error": getattr(exc, "error_type", "MLOpsError"),
            "message": str(exc),
            "details": getattr(exc, "details", {}),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """일반 예외 핸들러"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
        },
    )


# 라우터 등록
app.include_router(monitoring_router, prefix="/api/v1", tags=["monitoring"])
app.include_router(recommendations_router, prefix="/api/v1", tags=["recommendations"])
app.include_router(training_router, prefix="/api/v1", tags=["training"])


# 엔드포인트
@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus 메트릭 노출"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/health")
async def health_check() -> dict[str, str]:
    """헬스 체크"""
    return {"status": "healthy", "service": "mlops-recommendation-platform"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
