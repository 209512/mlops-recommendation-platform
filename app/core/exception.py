import logging
from typing import Any

from fastapi import HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    """표준화된 에러 응답 모델"""

    status: str = "error"
    message: str
    error_code: str | None = None
    details: dict[str, Any] | None = None


class MLOpsError(Exception):
    """MLOps 시스템 기본 예외 클래스"""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        logger.error(f"{self.__class__.__name__}: {message}")
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 형태로 변환"""
        return {
            "error": self.error_code or "InternalServerError",
            "message": self.message,
            "details": self.details,
        }

    def to_http_exception(self) -> HTTPException:
        """MLOpsError를 HTTPException으로 변환"""
        return HTTPException(
            status_code=self.status_code,
            detail=ErrorResponse(
                message=self.message, error_code=self.error_code, details=self.details
            ).model_dump(),
        )


class ModelNotFoundError(MLOpsError):
    """모델을 찾을 수 없을 때 발생하는 예외"""

    def __init__(self, model_name: str, version: str | None = None):
        message = f"Model '{model_name}'"
        if version:
            message += f" version '{version}'"
        message += " not found"
        super().__init__(
            message=message, error_code="MODEL_NOT_FOUND", status_code=status.HTTP_404_NOT_FOUND
        )


class TrainingError(MLOpsError):
    """모델 학습 중 발생하는 예외"""

    def __init__(self, message: str, model_type: str | None = None):
        full_message = "Training error"
        if model_type:
            full_message += f" in {model_type}"
        full_message += f": {message}"
        super().__init__(
            message=full_message,
            error_code="TRAINING_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


class DataLoadError(MLOpsError):
    """데이터 로딩 중 발생하는 예외"""

    def __init__(self, message: str, source: str | None = None):
        full_message = "Data loading error"
        if source:
            full_message += f" from {source}"
        full_message += f": {message}"
        super().__init__(
            message=full_message,
            error_code="DATA_LOAD_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class RecommendationError(MLOpsError):
    """추천 생성 중 발생하는 예외"""

    def __init__(self, message: str, user_id: int | None = None):
        full_message = "Recommendation error"
        if user_id:
            full_message += f" for user {user_id}"
        full_message += f": {message}"
        super().__init__(
            message=full_message,
            error_code="RECOMMENDATION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


class MLflowServiceError(MLOpsError):
    """MLflow 서비스 관련 예외"""

    def __init__(self, message: str, service: str | None = None):
        full_message = "MLflow service error"
        if service:
            full_message += f" in {service}"
        full_message += f": {message}"
        super().__init__(
            message=full_message,
            error_code="MLFLOW_SERVICE_ERROR",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class MLflowTrackingError(MLOpsError):
    """MLflow 추적 관련 예외"""

    def __init__(self, message: str, run_id: str | None = None):
        full_message = "MLflow tracking error"
        if run_id:
            full_message += f" for run {run_id}"
        full_message += f": {message}"
        super().__init__(
            message=full_message,
            error_code="MLFLOW_TRACKING_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
        )


def create_http_exception(
    status_code: int, detail: str, error_code: str | None = None
) -> HTTPException:
    """HTTPException 생성 헬퍼 함수"""

    content = {"detail": detail}
    if error_code:
        content["error_code"] = error_code

    return HTTPException(status_code=status_code, detail=content)
