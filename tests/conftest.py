import os
import sys
import tempfile
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.security import SecurityManager
from app.infrastructure.redis import RedisClient

# Global MLflow mock setup
mock_mlflow_client = MagicMock()
patcher = patch("mlflow.MlflowClient", return_value=mock_mlflow_client)
patcher.start()


@pytest.fixture(scope="session", autouse=True)
def clear_app_cache() -> Generator[None, None, None]:
    """애플리케이션 모듈 캐시 정리"""
    modules_to_remove = [mod for mod in sys.modules.keys() if mod.startswith("app")]
    for mod in modules_to_remove:
        del sys.modules[mod]
    yield
    modules_to_remove = [mod for mod in sys.modules.keys() if mod.startswith("app")]
    for mod in modules_to_remove:
        del sys.modules[mod]


@pytest.fixture
def test_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Factory pattern을 사용한 완전히 격리된 FastAPI 테스트 클라이언트"""
    # 환경 변수 설정
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-experiment")
    monkeypatch.setenv("DEBUG", "true")

    # 모듈 캐시 정리 후 새로운 앱 생성
    modules_to_remove = [mod for mod in sys.modules.keys() if mod.startswith("app")]
    for mod in modules_to_remove:
        del sys.modules[mod]

    from app.main import create_app

    app = create_app()

    return TestClient(app)


@pytest.fixture
def test_settings() -> Settings:
    """테스트용 Settings 객체 생성"""
    return Settings(
        environment="test",
        mlflow_tracking_uri="http://localhost:5001",
        database_url="sqlite+aiosqlite:///:memory:",
        redis_url="redis://localhost:6379/1",
        secret_key="test-secret-key",
        mlflow_experiment_name="test-experiment",
        debug=True,
    )


@pytest.fixture
def test_app(test_settings: Settings) -> TestClient:
    """Factory pattern을 사용한 테스트용 FastAPI 앱"""
    with patch("app.core.config.settings", test_settings):
        from app.main import create_app

        app = create_app()
        return TestClient(app)


@pytest.fixture
async def mock_db() -> AsyncMock:
    """Mock database session"""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_repositories(mock_db: AsyncSession) -> dict[str, Any]:
    """Mock all repositories"""
    return {
        "lecture_repo": AsyncMock(),
        "user_repo": AsyncMock(),
        "bookmark_repo": AsyncMock(),
        "search_log_repo": AsyncMock(),
        "user_pref_repo": AsyncMock(),
    }


@pytest.fixture
def reset_redis_client() -> Generator[None, None, None]:
    """Reset Redis client singleton for test isolation"""
    original_instance = getattr(RedisClient, "_instance", None)
    original_client = getattr(RedisClient, "_client", None)
    RedisClient._instance = None
    RedisClient._client = None

    yield

    RedisClient._instance = original_instance
    RedisClient._client = original_client


@pytest.fixture
def mock_redis_client() -> MagicMock:
    """Mock Redis client for testing without actual Redis connection"""
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.aclose = AsyncMock()
    return mock_client


@pytest.fixture
def sample_user_item_matrix() -> csr_matrix:
    """테스트용 사용자-아이템 행렬"""
    data = np.array([1, 1, 1, 1])
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    return csr_matrix((data, (rows, cols)), shape=(2, 2))


@pytest.fixture
def mock_lecture() -> MagicMock:
    """Mock lecture 객체"""
    lecture = MagicMock()
    lecture.id = 1
    lecture.title = "Test Lecture"
    lecture.instructor = "Test Instructor"
    lecture.thumbnail_img_url = "http://example.com/thumb.jpg"
    lecture.platform = "test-platform"
    lecture.difficulty = "초급"
    lecture.original_price = 10000
    lecture.discount_price = 8000
    lecture.rating = 4.5
    lecture.review_count = 100
    lecture.categories = [MagicMock(name="Test Category")]
    lecture.uuid = "test-uuid"
    lecture.description = "Test description"
    lecture.url_link = "http://example.com/lecture"
    lecture.is_active = True
    return lecture


@pytest.fixture
def security_manager() -> SecurityManager:
    """Security manager fixture"""
    return SecurityManager()


@pytest.fixture
def mock_mlflow_client_fixture() -> Generator[MagicMock, None, None]:
    """Mock MLflow client for test isolation"""
    with patch("mlflow.MlflowClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture(autouse=True)
def mock_mlflow_client_global() -> Generator[MagicMock, None, None]:
    """모든 테스트에 자동으로 MLflow 클라이언트 mock 적용"""
    with patch("mlflow.MlflowClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_experiment() -> MagicMock:
    """Mock MLflow experiment"""
    experiment = MagicMock()
    experiment.experiment_id = "test-experiment-id"
    experiment.name = "test-experiment"
    return experiment


@pytest.fixture
def mock_run() -> MagicMock:
    """Mock MLflow run"""
    run = MagicMock()
    run.info.run_id = "test-run-id"
    run.info.status = "FINISHED"
    run.data.metrics = {"accuracy": 0.95}
    return run


@pytest.fixture
def sample_sklearn_model() -> LogisticRegression:
    """Sample sklearn model for testing"""
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression()
    model.fit(x, y)
    return model


@pytest.fixture
def temp_file() -> Generator[str, None, None]:
    """임시 파일 경로 fixture"""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture(scope="session", autouse=True)
def cleanup_mlflow_patch() -> Generator[None, None, None]:
    """Clean up global MLflow patch after test session"""
    yield
    patcher.stop()


@pytest.fixture(scope="session", autouse=True)
def setup_prometheus_multiproc_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[None, None, None]:
    """Prometheus 멀티프로세스 디렉토리 설정"""
    import tempfile

    temp_dir = tempfile.mkdtemp()
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = temp_dir
    yield
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def clean_prometheus_registry() -> None:
    """각 테스트 전에 Prometheus 레지스트리 정리"""
    from prometheus_client import REGISTRY

    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
