import logging
from typing import Any

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """MLflow 모델 레지스트리 관리자"""

    def __init__(self, client: MlflowClient | None = None) -> None:
        self.client = client or MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
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

        Raises:
            MlflowException: 모델 등록 실패 시
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

        except MlflowException:
            # MLflow 예외는 그대로 전파
            raise
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise MlflowException(f"Failed to register model '{name}': {str(e)}") from e  # type: ignore

    def transition_model_stage(self, name: str, version: str, stage: str) -> bool:
        """
        모델 스테이지 전환

        Args:
            name: 모델 이름
            version: 모델 버전
            stage: 전환할 스테이지 (Staging, Production, Archived)

        Returns:
            성공 여부

        Raises:
            MlflowException: 스테이지 전환 실패 시
        """
        try:
            self.client.transition_model_version_stage(name=name, version=version, stage=stage)

            logger.info(f"Model {name} v{version} transitioned to {stage}")
            return True

        except MlflowException:
            # MLflow 예외는 그대로 전파
            raise
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            raise MlflowException(
                f"Failed to transition model '{name}' v{version} to '{stage}': {str(e)}"
            ) from e  # type: ignore

    def get_model_versions(self, name: str) -> list[dict[str, Any]]:
        """
        모델 버전 목록 조회

        Args:
            name: 모델 이름

        Returns:
            모델 버전 목록 (실패 시 빈 리스트)
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
            # 조회 실패 시 빈 리스트 반환 (호출자가 예외 처리하지 않아도 안전)
            return []

    def get_production_model(self, name: str) -> str | None:
        """
        프로덕션 모델 조회

        Args:
            name: 모델 이름

        Returns:
            프로덕션 모델 URI (없거나 실패 시 None)
        """
        try:
            model_versions = self.client.get_latest_versions(name=name, stages=["Production"])

            if model_versions:
                latest_version = model_versions[0]
                return f"models:/{name}/{latest_version.version}"

            return None

        except Exception as e:
            logger.error(f"Failed to get production model: {e}")
            # 조회 실패 시 None 반환 (호출자가 예외 처리하지 않아도 안전)
            return None

    def delete_model_version(self, name: str, version: str) -> bool:
        """
        모델 버전 삭제

        Args:
            name: 모델 이름
            version: 모델 버전

        Returns:
            성공 여부

        Raises:
            MlflowException: 모델 삭제 실패 시
        """
        try:
            self.client.delete_model_version(name=name, version=version)
            logger.info(f"Model {name} v{version} deleted")
            return True

        except MlflowException:
            # MLflow 예외는 그대로 전파
            raise
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            raise MlflowException(f"Failed to delete model '{name}' v{version}: {str(e)}") from e  # type: ignore

    def archive_model(self, name: str, version: str) -> bool:
        """
        모델 아카이빙

        Args:
            name: 모델 이름
            version: 모델 버전

        Returns:
            성공 여부

        Raises:
            MlflowException: 아카이빙 실패 시
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

        Raises:
            MlflowException: 승격 실패 시
        """
        return self.transition_model_stage(name, version, "Production")
