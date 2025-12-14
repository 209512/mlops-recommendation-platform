import logging
from typing import Literal

import boto3
from mypy_boto3_cloudwatch import CloudWatchClient
from mypy_boto3_ecr import ECRClient
from mypy_boto3_s3 import S3Client

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
        self.s3_client: S3Client = self.session.client("s3")

        # ECR 클라이언트
        self.ecr_client: ECRClient = self.session.client("ecr")

        # CloudWatch 클라이언트
        self.cloudwatch_client: CloudWatchClient = self.session.client("cloudwatch")

    def upload_to_s3(self, file_path: str, bucket: str, key: str) -> bool:
        """S3에 파일 업로드"""
        try:
            self.s3_client.upload_file(file_path, bucket, key)
            return True
        except Exception as e:
            logger.error(f"S3 upload error: {e}")
            return False

    def download_from_s3(self, bucket: str, key: str, file_path: str) -> bool:
        """S3에서 파일 다운로드"""
        try:
            self.s3_client.download_file(bucket, key, file_path)
            return True
        except Exception as e:
            logger.error(f"S3 download error: {e}")
            return False

    def put_cloudwatch_metric(
        self,
        namespace: str,
        metric_name: str,
        value: float,
        unit: Literal[
            "Bits",
            "Bits/Second",
            "Bytes",
            "Bytes/Second",
            "Count",
            "Count/Second",
            "Gigabits",
            "Gigabits/Second",
            "Gigabytes",
            "Gigabytes/Second",
            "Kilobits",
            "Kilobits/Second",
            "Kilobytes",
            "Kilobytes/Second",
            "Megabits",
            "Megabits/Second",
            "Megabytes",
            "Megabytes/Second",
            "Microseconds",
            "Milliseconds",
            "None",
            "Percent",
            "Seconds",
            "Terabits",
            "Terabits/Second",
            "Terabytes",
            "Terabytes/Second",
        ] = "Count",
    ) -> bool:
        """CloudWatch에 메트릭 전송"""
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


def get_s3_client() -> S3Client:
    """S3 클라이언트 반환"""
    return aws_client.s3_client
