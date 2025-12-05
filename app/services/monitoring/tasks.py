import logging
from typing import Any

from celery import current_app

from app.services.monitoring.prometheus import get_monitoring_service

logger = logging.getLogger(__name__)


@current_app.task(bind=True, name="monitor_model_performance")
def monitor_model_performance(self: Any) -> dict[str, Any]:
    """모델 성능 모니터링 태스크"""
    try:
        monitoring = get_monitoring_service()
        monitoring.update_system_metrics()
        monitoring.update_mlflow_metrics()

        return {"status": "success", "message": "Model performance monitored"}
    except Exception as e:
        logger.error(f"Failed to monitor model performance: {e}")
        return {"status": "error", "error": str(e)}
