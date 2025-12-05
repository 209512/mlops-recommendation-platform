from collections.abc import Callable
from typing import Any, TypeVar

from celery import Celery
from celery.schedules import crontab

from app.core.config import settings

# TypeVar for generic function types
F = TypeVar("F", bound=Callable[..., Any])

# Celery 애플리케이션 생성
celery_app = Celery(
    "mlops_recommendation",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.services.recommendation.tasks",
        "app.services.mlflow.tasks",
        "app.services.monitoring.tasks",  # 추가
    ],
)

# Celery 설정
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30분
    task_soft_time_limit=25 * 60,  # 25분
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# 주기적 태스크 설정
celery_app.conf.beat_schedule = {
    # 매일 새벽 2시에 전체 모델 학습
    "train-als-model-daily": {
        "task": "train_als_model",  # 실제 태스크 이름으로 수정
        "schedule": crontab(hour=2, minute=0),
    },
    # 매시간 모델 성능 모니터링
    "monitor-model-hourly": {
        "task": "monitor_model_performance",  # monitoring.tasks의 태스크
        "schedule": crontab(minute=0),
    },
    # 매일 새벽 3시에 사용자 선호도 업데이트
    "update-user-preferences-daily": {
        "task": "update_user_preferences",  # 실제 태스크 이름으로 수정
        "schedule": crontab(hour=3, minute=0),
    },
}


def celery_task(task_name: str) -> Callable[[F], F]:
    """
    Celery 태스크 데코레이터

    Args:
        task_name: 태스크 이름
    """

    def decorator(func: F) -> F:
        return celery_app.task(name=task_name, bind=True)(func)  # type: ignore

    return decorator


def check_celery_health() -> dict[str, Any]:
    """
    Celery 헬스 체크

    Returns:
        헬스 상태 정보
    """
    try:
        inspect = celery_app.control.inspect()
        stats = inspect.stats()

        if stats:
            active_workers = len(stats)
            return {"status": "healthy", "active_workers": active_workers, "worker_stats": stats}
        else:
            return {"status": "unhealthy", "message": "No active workers"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
