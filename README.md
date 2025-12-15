# MLOps Recommendation Platform

A production-ready FastAPI-based ALS recommendation system with a complete MLOps pipeline.

## 🚀 Key Features

- **ALS-based Recommendation System**: Collaborative filtering using the Implicit library
- **Complete MLOps Pipeline**: MLflow experiment tracking, Kubeflow pipeline automation
- **Real-time Monitoring**: Prometheus + Grafana dashboards
- **Asynchronous Processing**: Celery for background tasks
- **Containerization**: Multi-stage Docker builds
- **Cloud Deployment**: Kubernetes deployment with HPA auto-scaling

## 📋 System Requirements

- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+
- Kubernetes (for production deployment)

## 🛠️ Tech Stack

### Backend
- **FastAPI**: Web framework
- **SQLAlchemy 2.0**: ORM
- **Pydantic**: Data validation
- **Alembic**: Database migrations

### Machine Learning
- **Implicit**: ALS recommendation algorithm
- **SciPy**: Numerical computing
- **NumPy**: Array processing
- **MLflow**: Experiment tracking and model registry

### Infrastructure
- **PostgreSQL**: Primary database
- **Redis**: Cache and Celery broker
- **Celery**: Asynchronous task queue
- **AWS S3**: Model artifact storage

### Monitoring
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Prometheus Client**: Application metrics

### Deployment
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **Kubeflow**: ML pipelines
- **AWS ECR**: Container registry

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/209512/mlops-recommendation-platform.git
cd mlops-recommendation-platform
```

2. **Set up uv virtual environment**
```bash
# Install uv if not already installed
pip install uv

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"
```

3. **Environment configuration**
```bash
cp .env.example .env
# Configure necessary values in .env file
```

4. **Start services with Docker Compose**
```bash
docker-compose up -d
```

5. **Verify applications**
    - API: http://localhost:8000
    - API Documentation: http://localhost:8000/docs
    - MLflow: http://localhost:5001
    - Grafana: http://localhost:3000
    - Prometheus: http://localhost:9090

### Development Setup

```bash
# Code formatting
black app/ tests/
ruff check app/ tests/

# Type checking
mypy app/

# Run tests
pytest --cov=app
```

## 📁 Project Structure

```
mlops-recommendation-platform/
├── app/                          # Application source
│   ├── api/                      # API routes
│   │   ├── v1/                   # API v1 version
│   │   │   ├── recommendations.py
│   │   │   ├── training.py
│   │   │   ├── mlflow.py
│   │   │   └── monitoring.py
│   │   ├── dependencies.py       # Dependency injection
│   │   └── middleware.py         # Middleware
│   ├── core/                     # Core settings
│   │   ├── config.py             # Configuration management
│   │   ├── security.py           # Authentication/Security
│   │   └── exception.py          # Exception handling
│   ├── infrastructure/           # Infrastructure layer
│   │   ├── database.py           # Database
│   │   ├── redis.py              # Redis client
│   │   ├── celery.py             # Celery setup
│   │   ├── aws.py                # AWS integration
│   │   └── mlflow_server/        # MLflow server
│   ├── models/                   # Database models
│   ├── schemas/                  # Pydantic schemas
│   ├── services/                 # Business logic
│   │   ├── recommendation/       # Recommendation service
│   │   │   ├── repositories/     # Data access layer
│   │   │   ├── service.py        # Main recommendation logic
│   │   │   ├── data_loader.py    # Data loading
│   │   │   └── model_trainer.py  # Model training
│   │   ├── mlflow/               # MLflow services
│   │   └── monitoring/           # Monitoring services
│   └── main.py                   # Application entry point
├── k8s/                          # Kubernetes manifests
│   ├── deployment.yaml           # Deployment configuration
│   ├── hpa.yaml                  # Horizontal Pod Autoscaler
│   ├── service.yaml              # Service configuration
│   └── kubeflow/                 # Kubeflow pipelines
├── monitoring/                   # Monitoring setup
│   ├── prometheus/               # Prometheus configuration
│   └── grafana/                  # Grafana dashboards
├── scripts/                      # Utility scripts
├── tests/                        # Test suite
└── pyproject.toml               # Project configuration
```

## 🔧 API Usage

### Get Recommendations
```bash
curl -X GET "http://localhost:8000/api/v1/recommendations/user/{user_id}?limit=10"
```

### Train Model
```bash
curl -X POST "http://localhost:8000/api/v1/training/train" \
  -H "Content-Type: application/json" \
  -d '{"model_type": "als", "parameters": {"factors": 100, "iterations": 50}}'
```

### Evaluate Model
```bash
curl -X GET "http://localhost:8000/api/v1/training/evaluate/{run_id}"
```

## 📊 Monitoring

### Key Metrics
- **Recommendation Latency**: `recommendation_latency_seconds`
- **Model Training Time**: `ml_training_duration_seconds`
- **Model Accuracy**: `ml_model_accuracy`
- **API Request Count**: `http_requests_total`

### Grafana Dashboards
- MLOps Comprehensive Dashboard: Model performance and system status
- Recommendation System Dashboard: Recommendation quality and user behavior

## 🚀 Production Deployment

### Kubernetes Deployment
```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Apply configuration and secrets
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

### CI/CD Pipeline
- **CI**: GitHub Actions for testing and building  
- **CD**: EKS deployment via ArgoCD  
- **Model Retraining**: Automated periodic retraining pipeline 

## 🏗️ Architecture & Design Highlights

### Recommendation Architecture
**Multi-stage Fallback Strategy**: ALS-based collaborative filtering with intelligent fallback mechanisms to handle cold-start and data sparsity issues. When ALS recommendations fail, the system gracefully degrades to category-based recommendations, then to popular lectures, ensuring service availability while maintaining recommendation quality. 

### Performance Optimization
**Asynchronous Processing**: CPU-intensive ALS computations executed via `asyncio.to_thread()` to prevent blocking the main event loop, maintaining response times in the hundreds of milliseconds range even under load. Redis-based caching layer with user-specific invalidation significantly reduces latency for repeated requests. 

### Caching Strategy
**Intelligent Cache Management**: Redis-based user-specific recommendation caching with 5-minute TTL. Cache invalidation triggered by user interaction events ensures data freshness while optimizing performance for repeated requests.

### Reliability & Scalability
**Celery Task Architecture**: Background tasks with exponential backoff retry logic and structured logging maintain low failure rates. Process pool executor handles concurrent model training operations efficiently without memory leaks.

### MLOps Pipeline
**Automated Model Management**: Weekly retraining pipeline with performance gates ensures only models meeting quality thresholds are automatically promoted to production. MLflow integration provides complete experiment tracking and model versioning with rollback capability.

### Security & Monitoring
**Environment-based Security**: Production environment enforces short token expiration, HTTPS-only connections, and account lockout policies. Prometheus metrics collection with Grafana dashboards provides real-time visibility into system health and model performance.

## 🧪 Testing

### Test Execution
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/unit/test_recommendations.py

# Run with specific markers
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Run only integration tests
```

### Test Structure
The test suite is organized into multiple layers:

```
tests/
├── unit/                         # Unit tests
│   ├── test_recommendations.py   # Recommendation logic tests
│   ├── test_models.py           # Model tests
│   └── test_services.py         # Service layer tests
├── integration/                 # Integration tests
│   ├── test_api_endpoints.py    # API integration tests
│   ├── test_mlflow_integration.py # MLflow integration tests
│   └── test_celery_tasks.py     # Celery task tests
├── performance/                 # Performance tests
│   ├── test_recommendation_latency.py
│   └── test_model_training_performance.py
└── e2e/                        # End-to-end tests
    ├── test_full_recommendation_pipeline.py
    └── test_model_deployment_pipeline.py
```

### Test Configuration
Tests are configured in `pyproject.toml`  and use:
- **pytest**: Primary testing framework
- **pytest-asyncio**: For async test support
- **pytest-cov**: Coverage reporting
- **httpx**: For async HTTP client testing
- **aiosqlite**: For database testing

### Coverage Requirements
- Minimum coverage: 80% (enforced in CI) 
- Coverage reports generated in XML and HTML formats
- Integration with SonarQube for quality gates

### Test Best Practices
1. **Mock External Dependencies**: Use fixtures to mock MLflow, Redis, and AWS services
2. **Database Isolation**: Each test uses a clean database state
3. **Async Testing**: Proper async/await patterns for FastAPI endpoints
4. **Performance Benchmarks**: Track recommendation latency and model training time

## 💡 Technical Challenges & Solutions

### Challenge 1: Cold Start Handling for New Users
- **Problem**: Users without interaction history receive no meaningful recommendations
- **Solution**: Hybrid recommendation approach combining collaborative filtering with content-based features using user preferences and lecture categories
- **Result**: Improved user engagement for new users through personalized initial recommendations

### Challenge 2: Meeting Real-time Recommendation SLO
- **Problem**: ALS model inference taking multiple seconds causing poor user experience
- **Solution**: Asynchronous processing with `asyncio.to_thread()`, Redis caching layer, and model optimization techniques
- **Result**: Response times maintained in hundreds of milliseconds range even under load

### Challenge 3: Model Performance Degradation
- **Problem**: Recommendation quality declining over time as user preferences evolve
- **Solution**: Automated weekly retraining pipeline with performance gates and A/B testing before deployment
- **Result**: Consistent recommendation quality maintained through continuous model improvement 

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow the existing code style (Black + Ruff)
- Add tests for new features
- Update documentation
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

- Issues: [GitHub Issues](https://github.com/209512/mlops-recommendation-platform/issues)
- Documentation: [Project Wiki](https://github.com/209512/mlops-recommendation-platform/wiki)
