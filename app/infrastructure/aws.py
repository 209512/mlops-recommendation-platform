import logging

import boto3

from app.core.config import settings

logger = logging.getLogger(__name__)


class AWSClient:
    """AWS 클라이언트 관리 클래스"""

    def __init__(self) -> None:
        self.session = boto3.Session(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )

        # S3 클라이언트
        self.s3_client = self.session.client("s3")

        # ECR 클라이언트
        self.ecr_client = self.session.client("ecr")

        # CloudWatch 클라이언트
        self.cloudwatch_client = self.session.client("cloudwatch")

    def upload_to_s3(self, file_path: str, bucket: str, key: str) -> bool:
        """
        S3에 파일 업로드

        Args:
            file_path: 로컬 파일 경로
            bucket: S3 버킷 이름
            key: S3 객체 키

        Returns:
            성공 여부
        """
        try:
            self.s3_client.upload_file(file_path, bucket, key)
            return True
        except Exception as e:
            logger.error(f"S3 upload error: {e}")
            return False

    def download_from_s3(self, bucket: str, key: str, file_path: str) -> bool:
        """
        S3에서 파일 다운로드

        Args:
            bucket: S3 버킷 이름
            key: S3 객체 키
            file_path: 로컬 저장 경로

        Returns:
            성공 여부
        """
        try:
            self.s3_client.download_file(bucket, key, file_path)
            return True
        except Exception as e:
            logger.error(f"S3 download error: {e}")
            return False

    def put_cloudwatch_metric(
        self, namespace: str, metric_name: str, value: float, unit: str = "Count"
    ) -> bool:
        """
        CloudWatch에 메트릭 전송

        Args:
            namespace: 메트릭 네임스페이스
            metric_name: 메트릭 이름
            value: 메트릭 값
            unit: 단위

        Returns:
            성공 여부
        """
        try:
            self.cloudwatch_client.put_metric_data(
                Namespace=namespace,
                MetricData=[{"MetricName": metric_name, "Value": value, "Unit": unit}],
            )
            return True
        except Exception as e:
            logger.error(f"CloudWatch metric error: {e}")
            return False


# AWS 클라이언트 인스턴스
aws_client = AWSClient()
