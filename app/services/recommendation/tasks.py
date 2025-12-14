import asyncio
import logging
from datetime import datetime
from typing import Any

import implicit
from celery import current_app
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from app.core.config import settings
from app.infrastructure.database import get_async_db
from app.services.mlflow.registry import ModelRegistry
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
    experiment_name: str = "recommendation_experiments",
) -> dict[str, Any]:
    """ALS 모델 학습 Celery 태스크"""
    run_id = None
    mlflow_service = None
    try:
        # 데이터 로더 초기를 위한 비동기 컨텍스트 사용
        async def _train_model() -> dict[str, Any]:
            db_gen = get_async_db()
            db = await db_gen.__anext__()
            try:
                # 리포지토리 초기화
                lecture_repo = LectureRepository(db)
                user_repo = UserRepository(db)
                bookmark_repo = BookmarkRepository(db)
                search_log_repo = SearchLogRepository(db)
                user_pref_repo = UserPreferenceRepository(db)

                # 데이터 로더 초기화
                data_loader = ALSDataLoader(
                    lecture_repo=lecture_repo,
                    user_repo=user_repo,
                    bookmark_repo=bookmark_repo,
                    search_log_repo=search_log_repo,
                    user_pref_repo=user_pref_repo,
                )

                # 사용자-아이템 행렬 로드
                matrix_bundle = await data_loader.load_training_data()
                if not matrix_bundle:
                    raise ValueError("No training data available")

                return matrix_bundle
            finally:
                await db_gen.aclose()

        matrix_bundle = asyncio.run(_train_model())
        user_item_matrix = matrix_bundle["matrix"]

        # ALS 모델 초기화 및 학습
        als_model = implicit.als.AlternatingLeastSquares(
            factors=getattr(settings, "als_factors", 100),
            regularization=getattr(settings, "als_regularization", 0.01),
            iterations=getattr(settings, "als_iterations", 15),
            calculate_training_loss=True,
        )

        als_model.fit(user_item_matrix)

        # MLflow 서비스 가져오기
        mlflow_service = get_mlflow_tracking_service()
        if mlflow_service:
            # 실험 생성
            experiment_id = mlflow_service.create_experiment(
                name=experiment_name, tags={"model_type": "als", "environment": "production"}
            )

            # 실행 시작
            run_id = mlflow_service.start_run(run_name=f"{model_name}_{datetime.now().isoformat()}")

            if run_id:
                try:
                    # 모델 로깅 (run_id와 model_name 파라미터 사용)
                    model_info = mlflow_service.log_model(
                        run_id=run_id, model=als_model, model_name=model_name, model_type="sklearn"
                    )

                    # run 종료 - 성공 상태로
                    mlflow_service.end_run(run_id=run_id, status="FINISHED")
                    logger.info(f"MLflow run {run_id} finished successfully")

                    # run_id 추출 (bool 타입 체크)
                    final_run_id = run_id if isinstance(run_id, str) else ""

                    return {
                        "status": "success",
                        "model_name": model_name,
                        "experiment_id": experiment_id,
                        "run_id": final_run_id,
                        "model_info": model_info,
                        "experiment": experiment_name,
                    }
                except Exception as e:
                    # 모델 로깅 중 에러 발생 시 run을 FAILED 상태로 종료
                    if run_id:
                        try:
                            mlflow_service.end_run(run_id=run_id, status="FAILED")
                            logger.error(f"MLflow run {run_id} failed: {str(e)}")
                        except Exception as end_error:
                            logger.error(f"Failed to end MLflow run {run_id}: {str(end_error)}")
                    raise
            else:
                return {
                    "status": "error",
                    "error": "Failed to start MLflow run",
                    "model_name": model_name,
                }
        else:
            return {
                "status": "error",
                "error": "MLflow service not available",
                "model_name": model_name,
            }

    except Exception as e:
        # 최상위 예외 처리 - run이 시작된 경우 FAILED 상태로 종료
        if run_id and mlflow_service:
            try:
                mlflow_service.end_run(run_id=run_id, status="FAILED")
                logger.error(f"MLflow run {run_id} failed due to exception: {str(e)}")
            except Exception as end_error:
                logger.error(f"Failed to end MLflow run {run_id}: {str(end_error)}")

        logger.error(f"Failed to train ALS model: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "model_name": model_name,
        }


@current_app.task(bind=True, name="generate_recommendations")
def generate_recommendations(
    self: Any, user_id: int, num_recommendations: int = 10
) -> dict[str, Any]:
    """사용자 추천 생성"""
    try:
        # 비동기 컨텍스트 매니저 사용
        async def _generate_recommendations() -> list[Any]:
            db_gen = get_async_db()
            db = await db_gen.__anext__()
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

                # 추천 생성 (메서드 이름 수정)
                recommendations = await recommendation_service.get_recommendations(
                    user_id=user_id, limit=num_recommendations
                )

                return recommendations
            finally:
                await db_gen.aclose()

                # 비동기 함수 실행

        recommendations = asyncio.run(_generate_recommendations())

        return {
            "status": "success",
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations) if recommendations else 0,
        }

    except Exception as e:
        logger.error(f"Failed to generate recommendations for user {user_id}: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "user_id": user_id,
        }


@current_app.task(bind=True, name="update_user_preferences")
def update_user_preferences(
    self: Any, user_id: int, category_id: int, weight: float = 1.0
) -> dict[str, Any]:
    """사용자 선호도 업데이트"""
    try:
        # 비동기 컨텍스트 매니저 사용
        async def _update_preferences() -> None:
            db_gen = get_async_db()
            db = await db_gen.__anext__()
            try:
                # 리포지토리 초기화
                user_pref_repo = UserPreferenceRepository(db)

                # 선호도 업데이트
                await user_pref_repo.update_category_preference(user_id, category_id, weight)
            finally:
                await db_gen.aclose()

                # 비동기 함수 실행

        asyncio.run(_update_preferences())

        return {
            "status": "success",
            "user_id": user_id,
            "category_id": category_id,
            "weight": weight,
        }

    except Exception as e:
        logger.error(f"Failed to update user preference: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "user_id": user_id,
        }


@current_app.task(bind=True, name="batch_recommendations")
def batch_recommendations(
    self: Any, user_ids: list[int], num_recommendations: int = 10
) -> dict[str, Any]:
    """배치 추천 생성"""
    try:
        results = []

        for user_id in user_ids:
            try:
                # 개별 추천 생성 태스크 호출
                result = generate_recommendations(user_id, num_recommendations)
                results.append(result)

            except Exception as e:
                error_msg = f"Failed to generate recommendations for user {user_id}: {str(e)}"
                logger.error(error_msg)
                results.append(
                    {
                        "status": "error",
                        "error": str(e),
                        "user_id": user_id,
                    }
                )

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
    models_to_keep: int = 5,
) -> dict[str, Any]:
    """오래된 모델 정리"""
    try:
        logger.info(
            f"Starting model cleanup: keeping models "
            f"from last {days_to_keep} days, minimum {models_to_keep} models"
        )

        # MLflow 모델 레지스트리 연동
        model_registry = ModelRegistry()
        if not model_registry.client:
            raise Exception("MLflow registry client not available")

        # 모델 이름 목록 가져오기
        model_name = "als_model"
        model_versions = model_registry.get_model_versions(model_name)

        if not model_versions:
            logger.info("No models found to cleanup")
            return {
                "status": "success",
                "days_to_keep": days_to_keep,
                "models_to_keep": models_to_keep,
                "cleaned_models": [],
                "total_models": 0,
            }

        # 정리할 모델 결정
        import datetime

        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)

        models_to_delete = []
        models_kept = []

        # 최신 모델부터 정렬
        sorted_versions = sorted(
            model_versions, key=lambda x: x.get("creation_timestamp", 0), reverse=True
        )

        # 최소 보유 개수는 유지
        recent_models = sorted_versions[:models_to_keep]
        older_models = sorted_versions[models_to_keep:]

        # 최신 모델들 중에서도 기간 내 모델만 유지
        for model in recent_models:
            creation_time = datetime.datetime.fromtimestamp(
                model.get("creation_timestamp", 0) / 1000
            )
            if creation_time >= cutoff_date:
                models_kept.append(model)
            else:
                models_to_delete.append(model)

        # 오래된 모델들 중에서도 기간 초과 모델만 삭제
        for model in older_models:
            creation_time = datetime.datetime.fromtimestamp(
                model.get("creation_timestamp", 0) / 1000
            )
            if creation_time < cutoff_date:
                models_to_delete.append(model)
            else:
                models_kept.append(model)

        # 모델 삭제 실행
        deleted_models = []
        for model in models_to_delete:
            try:
                version = model.get("version")
                if version:
                    # 모델을 Archived 상태로 변경 후 삭제
                    model_registry.archive_model(model_name, version)
                    model_registry.delete_model_version(model_name, version)
                    deleted_models.append(version)
                    logger.info(f"Deleted model version: {version}")
            except Exception as e:
                logger.warning(f"Failed to delete model version {model.get('version')}: {e}")

        # Prometheus 메트릭 기록
        try:
            prometheus_registry = CollectorRegistry()
            cleanup_gauge = Gauge(
                "model_cleanup_count", "Number of models cleaned up", registry=prometheus_registry
            )
            cleanup_gauge.set(len(deleted_models))

            push_to_gateway("localhost:9091", job="model_cleanup", registry=prometheus_registry)
        except Exception as e:
            logger.warning(f"Failed to push metrics: {e}")

        logger.info(
            f"Model cleanup completed: "
            f"deleted {len(deleted_models)} models, kept {len(models_kept)} models"
        )

        return {
            "status": "success",
            "days_to_keep": days_to_keep,
            "models_to_keep": models_to_keep,
            "cleaned_models": deleted_models,
            "total_models": len(model_versions),
            "kept_models": len(models_kept),
        }

    except Exception as e:
        logger.error(f"Failed to cleanup old models: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "days_to_keep": days_to_keep,
            "models_to_keep": models_to_keep,
            "cleaned_models": [],
        }
