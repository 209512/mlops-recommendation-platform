import logging
from typing import Any

from mlflow.tracking import MlflowClient

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """MLflow 모델 레지스트리 관리자"""

    def __init__(self) -> None:
        self.client = MlflowClient()
        self.registry_uri = settings.mlflow_tracking_uri

    def register_model(self, model_uri: str, name: str, description: str = "") -> dict[str, Any]:
        """
        모델 등록

        Args:
            model_uri: 모델 URI
            name: 모델 이름
            description: 모델 설명

        Returns:
            모델 버전 정보
        """
        try:
            # 모델 등록
            model_version = self.client.create_model_version(
                name=name, source=model_uri, description=description
            )

            logger.info(f"Model registered: {name} version {model_version.version}")
            return {
                "name": name,
                "version": model_version.version,
                "creation_timestamp": model_version.creation_timestamp,
                "description": description,
            }

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def transition_model_stage(self, name: str, version: str, stage: str) -> bool:
        """
        모델 스테이지 전환

        Args:
            name: 모델 이름
            version: 모델 버전
            stage: 전환할 스테이지 (Staging, Production, Archived)

        Returns:
            성공 여부
        """
        try:
            self.client.transition_model_version_stage(name=name, version=version, stage=stage)

            logger.info(f"Model {name} v{version} transitioned to {stage}")
            return True

        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            return False

    def get_model_versions(self, name: str) -> list[dict[str, Any]]:
        """
        모델 버전 목록 조회

        Args:
            name: 모델 이름

        Returns:
            모델 버전 목록
        """
        try:
            versions = self.client.search_model_versions(f"name='{name}'")

            result = []
            for version in versions:
                result.append(
                    {
                        "name": version.name,
                        "version": version.version,
                        "stage": version.current_stage,
                        "creation_timestamp": version.creation_timestamp,
                        "last_updated_timestamp": version.last_updated_timestamp,
                        "description": version.description or "",
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Failed to get model versions: {e}")
            return []

    def get_production_model(self, name: str) -> str | None:
        """
        프로덕션 모델 조회

        Args:
            name: 모델 이름

        Returns:
            프로덕션 모델 URI
        """
        try:
            model_versions = self.client.get_latest_versions(name=name, stages=["Production"])

            if model_versions:
                latest_version = model_versions[0]
                return f"models:/{name}/{latest_version.version}"

            return None

        except Exception as e:
            logger.error(f"Failed to get production model: {e}")
            return None

    def delete_model_version(self, name: str, version: str) -> bool:
        """
        모델 버전 삭제

        Args:
            name: 모델 이름
            version: 모델 버전

        Returns:
            성공 여부
        """
        try:
            self.client.delete_model_version(name=name, version=version)
            logger.info(f"Model {name} v{version} deleted")
            return True

        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            return False

    def archive_model(self, name: str, version: str) -> bool:
        """
        모델 아카이빙

        Args:
            name: 모델 이름
            version: 모델 버전

        Returns:
            성공 여부
        """
        return self.transition_model_stage(name, version, "Archived")

    def promote_to_production(self, name: str, version: str) -> bool:
        """
        프로덕션으로 승격

        Args:
            name: 모델 이름
            version: 모델 버전

        Returns:
            성공 여부
        """
        return self.transition_model_stage(name, version, "Production")
