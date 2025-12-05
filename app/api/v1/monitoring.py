from typing import Any

from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.dependencies import get_monitoring_service
from app.services.monitoring.prometheus import MLOpsMonitoring

router = APIRouter()


@router.get("/metrics")
async def get_metrics() -> Response:
    """Prometheus 메트릭 엔드포인트"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/health")
async def health_check() -> dict[str, str]:
    """헬스 체크"""
    return {"status": "healthy", "service": "mlops-recommendation-api"}


@router.get("/ready")
async def readiness_check(
    monitoring: MLOpsMonitoring = Depends(get_monitoring_service),
) -> dict[str, str]:
    """준비 상태 체크"""
    try:
        # 기본 서비스 상태 확인
        monitoring.check_health()
        return {"status": "ready", "service": "mlops-recommendation-api"}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}


@router.get("/system-info")
async def get_system_info(
    monitoring: MLOpsMonitoring = Depends(get_monitoring_service),
) -> dict[str, str]:
    """시스템 정보 조회"""
    try:
        info = monitoring.get_system_info()
        return info
    except Exception as e:
        return {"error": str(e)}


@router.get("/alerts")
async def get_active_alerts(
    monitoring: MLOpsMonitoring = Depends(get_monitoring_service),
) -> dict[str, Any]:
    """활성 알림 조회"""
    try:
        alerts = monitoring.get_active_alerts()
        return {"alerts": alerts}
    except Exception as e:
        return {"error": str(e)}
