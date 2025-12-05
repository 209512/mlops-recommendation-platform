from typing import Any

from fastapi import HTTPException


class MLOpsError(Exception):
    """MLOps 시스템 기본 예외 클래스"""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ModelNotFoundError(MLOpsError):
    """모델을 찾을 수 없을 때 발생하는 예외"""

    def __init__(self, model_name: str, version: str | None = None):
        message = f"Model '{model_name}'"
        if version:
            message += f" version '{version}'"
        message += " not found"
        super().__init__(message, error_code="MODEL_NOT_FOUND")


class TrainingError(MLOpsError):
    """모델 학습 중 발생하는 예외"""

    def __init__(self, message: str, model_type: str | None = None):
        full_message = "Training error"
        if model_type:
            full_message += f" in {model_type}"
        full_message += f": {message}"
        super().__init__(full_message, error_code="TRAINING_ERROR")


class DataLoadError(MLOpsError):
    """데이터 로딩 중 발생하는 예외"""

    def __init__(self, message: str, source: str | None = None):
        full_message = "Data loading error"
        if source:
            full_message += f" from {source}"
        full_message += f": {message}"
        super().__init__(full_message, error_code="DATA_LOAD_ERROR")


class RecommendationError(MLOpsError):
    """추천 생성 중 발생하는 예외"""

    def __init__(self, message: str, user_id: int | None = None):
        full_message = "Recommendation error"
        if user_id:
            full_message += f" for user {user_id}"
        full_message += f": {message}"
        super().__init__(full_message, error_code="RECOMMENDATION_ERROR")


def create_http_exception(
    status_code: int, detail: str, error_code: str | None = None
) -> HTTPException:
    """HTTPException 생성 헬퍼 함수"""

    content = {"detail": detail}
    if error_code:
        content["error_code"] = error_code

    return HTTPException(status_code=status_code, detail=content)
