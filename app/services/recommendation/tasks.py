import logging
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.sparse import csr_matrix
import implicit

from celery import current_app

from app.core.config import settings
from app.infrastructure.database import get_async_db
from app.services.mlflow.tracking import get_mlflow_tracking_service
from app.services.recommendation.data_loader import ALSDataLoader
from app.services.recommendation.repositories import (
    BookmarkRepository,
    LectureRepository,
    SearchLogRepository,
    UserPreferenceRepository,
    UserRepository,
)
from app.services.recommendation.service import RecommendationService

logger = logging.getLogger(__name__)


@current_app.task(bind=True, name="train_als_model")
def train_als_model(
        self: Any,
        model_name: str = "als_recommendation_model",
        experiment_name: str = "recommendation_experiments"
) -> Dict[str, Any]:
    """ALS 모델 학습 Celery 태스크"""
    try:
        # 데이터 로더 초기화 (db 인자 제거)
        data_loader = ALSDataLoader()

        # 사용자-아이템 행렬 로드 (load_matrix 메서드 사용)
        user_item_matrix = data_loader.load_matrix()

        # ALS 모델 초기화 및 학습
        als_model = implicit.als.AlternatingLeastSquares(
            factors=getattr(settings, 'als_factors', 100),
            regularization=getattr(settings, 'als_regularization', 0.01),
            iterations=getattr(settings, 'als_iterations', 15),
            calculate_training_loss=True
        )

        als_model.fit(user_item_matrix)

        # MLflow 서비스 가져오기
        mlflow_service = get_mlflow_tracking_service()
        if mlflow_service:
            # 실험 생성 (MLflowTrackingServer 메서드 사용)
            experiment_id = mlflow_service.create_experiment(
                name=experiment_name,
                tags={"model_type": "als", "environment": "production"}
            )

            # 모델 로깅 (run_name과 artifact_path 제거)
            model_info = mlflow_service.log_model(
                model=als_model,
                model_path=f"{model_name}_{datetime.now().isoformat()}"
            )

            return {
                "status": "success",
                "model_name": model_name,
                "experiment_id": experiment_id,
                "run_id": model_info if isinstance(model_info, str) else model_info.get("run_id", ""),
                "experiment": experiment_name,
            }
        else:
            return {
                "status": "error",
                "error": "MLflow service not available",
                "model_name": model_name,
            }

    except Exception as e:
        logger.error(f"Failed to train ALS model: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "model_name": model_name,
        }


@current_app.task(bind=True, name="generate_recommendations")
def generate_recommendations(
        self: Any,
        user_id: int,
        num_recommendations: int = 10
) -> Dict[str, Any]:
    """사용자 추천 생성"""
    try:
        # 데이터베이스 세션 생성
        db_gen = get_async_db()
        db = next(db_gen)

        try:
            # 리포지토리 초기화
            lecture_repo = LectureRepository(db)
            user_repo = UserRepository(db)
            bookmark_repo = BookmarkRepository(db)
            search_log_repo = SearchLogRepository(db)
            user_pref_repo = UserPreferenceRepository(db)

            # 추천 서비스 초기화
            recommendation_service = RecommendationService(
                lecture_repo=lecture_repo,
                user_repo=user_repo,
                bookmark_repo=bookmark_repo,
                search_log_repo=search_log_repo,
                user_pref_repo=user_pref_repo,
            )

            # 추천 생성 (await 사용)
            recommendations = asyncio.run(
                recommendation_service.get_recommendations_for_user(
                    user_id=user_id,
                    limit=num_recommendations
                )
            )

            return {
                "status": "success",
                "user_id": user_id,
                "recommendations": recommendations,
                "count": len(recommendations) if recommendations else 0,
            }

        finally:
            # 데이터베이스 세션 정리
            if hasattr(db, 'close'):
                db.close()

    except Exception as e:
        logger.error(f"Failed to generate recommendations for user {user_id}: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "user_id": user_id,
        }


@current_app.task(bind=True, name="update_user_preferences")
def update_user_preferences(
        self: Any,
        user_id: int,
        category_id: int,
        weight: float = 1.0
) -> Dict[str, Any]:
    """사용자 선호도 업데이트"""
    try:
        # 데이터베이스 세션 생성
        db_gen = get_async_db()
        db = next(db_gen)

        try:
            # 리포지토리 초기화
            user_pref_repo = UserPreferenceRepository(db)

            # 선호도 업데이트 (await 사용, 변수 할당 제거)
            asyncio.run(
                user_pref_repo.update_category_preference(user_id, category_id, weight)
            )

            return {
                "status": "success",
                "user_id": user_id,
                "category_id": category_id,
                "weight": weight,
            }

        finally:
            # 데이터베이스 세션 정리
            if hasattr(db, 'close'):
                db.close()

    except Exception as e:
        logger.error(f"Failed to update user preference: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "user_id": user_id,
        }


@current_app.task(bind=True, name="batch_recommendations")
def batch_recommendations(
        self: Any,
        user_ids: List[int],
        num_recommendations: int = 10
) -> Dict[str, Any]:
    """배치 추천 생성"""
    try:
        results = []

        for user_id in user_ids:
            try:
                # 개별 추천 생성 태스크 호출
                result = generate_recommendations(user_id, num_recommendations)
                results.append(result)

            except Exception as e:
                # 긴 라인 분리하여 100자 제한 준수
                error_msg = (
                    f"Failed to generate recommendations for user {user_id}: {str(e)}"
                )
                logger.error(error_msg)
                results.append({
                    "status": "error",
                    "error": str(e),
                    "user_id": user_id,
                })

        return {
            "status": "success",
            "total_users": len(user_ids),
            "successful": sum(1 for r in results if r.get("status") == "success"),
            "results": results,
        }

    except Exception as e:
        logger.error(f"Failed to generate batch recommendations: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "user_ids": user_ids,
        }


@current_app.task(bind=True, name="cleanup_old_models")
def cleanup_old_models(
        self: Any,
        days_to_keep: int = 30,
) -> Dict[str, Any]:
    """오래된 모델 정리"""
    try:
        # TODO: MLflow 모델 정리 기능 구현
        logger.info(f"Cleaning up models older than {days_to_keep} days")

        return {
            "status": "success",
            "days_to_keep": days_to_keep,
            "cleaned_models": [],  # TODO: 실제 정리된 모델 목록으로 채우기
        }

    except Exception as e:
        logger.error(f"Failed to cleanup old models: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "days_to_keep": days_to_keep,
        }