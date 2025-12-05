Here is the English translation of the provided Korean README content. 🌍

# MLOps Recommendation Platform

A **production-ready, enterprise-grade MLOps recommendation system** that implements a high-performance recommendation API service based on **FastAPI**, utilizes the **implicit** library for the ALS algorithm, and provides a complete MLOps pipeline.

-----

## 🏗️ Architecture

The project follows **Clean Architecture** principles and applies clear separation of concerns:

```
app/
├── api/                   # REST API Endpoints
│   ├── dependencies.py    # Dependency Injection (with lazy loading)
│   ├── middleware.py      # Custom Middleware
│   └── v1/                # API Version 1
│       ├── recommendations.py # Recommendation Endpoints
│       ├── training.py        # Model Training Endpoints
│       └── monitoring.py      # Health Check Endpoints
├── core/                  # Core Application Logic
│   ├── config.py          # Environment Configuration
│   ├── exception.py       # Custom Exception Handling
│   └── security.py        # Security Utilities
├── models/                # SQLAlchemy Data Models
│   ├── base.py            # Base Model Class
│   ├── lecture.py         # Lecture Model (using Association Object Pattern)
│   └── user.py            # User Model
├── schemas/               # Pydantic Schemas
├── services/              # Business Logic
│   ├── recommendation/    # Recommendation Service
│   │   ├── service.py     # Recommendation Business Logic
│   │   ├── repositories/  # Data Access Layer
│   │   ├── data_loader.py # ALS Data Loader
│   │   ├── model_trainer.py # ALS Model Trainer
│   │   └── tasks.py       # Celery Background Tasks
│   └── mlflow/            # MLflow Service
│       ├── tracking.py    # Model Tracking Service
│       └── tasks.py       # MLflow Related Tasks
└── infrastructure/        # Infrastructure Layer
    ├── database.py        # Asynchronous Database Configuration
    ├── redis.py           # Redis Client
    └── celery.py          # Celery Configuration
```

-----

## 🛠️ Tech Stack

### Backend Framework

  - **FastAPI**: Modern, asynchronous web framework.
  - **SQLAlchemy 2.0**: Asynchronous ORM.
  - **Pydantic v2**: Data validation and serialization.

### Recommendation Algorithm

  - **Implicit Library**: Implementation of the **ALS (Alternating Least Squares)** algorithm.
  - **ALS (Alternating Least Squares)**: Collaborative Filtering for implicit feedback.

### MLOps Pipeline

  - **MLflow**: Model experiment tracking and management.
  - **Kubeflow**: ML workflow orchestration.
  - **Celery**: Distributed task queue for background processing.

### Database & Caching

  - **PostgreSQL**: Primary data store.
  - **Redis**: Caching and Celery broker.

### Monitoring & Logging

  - **Prometheus**: Metrics collection.
  - **Grafana**: Visualization dashboard.
  - **SonarQube**: Code quality analysis.

### Cloud & Deployment

  - **Kubernetes**: Container orchestration.
  - **AWS**: EKS, RDS, ElastiCache.
  - **Docker**: Containerization.

-----

## ⚙️ Configuration

The project manages configuration using environment variables via a `.env` file. Copy the `.env.example` file to `.env` and configure the necessary values.

### Required Environment Variables

**Database Settings**

```bash
DATABASE_URL="postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/mlops"
DB_HOST=db
DB_PORT=5432
DB_NAME=mlops
DB_USER=postgres
POSTGRES_PASSWORD=secure_password_change_in_production
```

**Redis Settings**

```bash
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_URL="redis://redis:6379/0"
```

**MLflow Settings**

```bash
MLFLOW_TRACKING_URI=http://mlflow:5001
MLFLOW_EXPERIMENT_NAME=als_recommendation
MLFLOW_BACKEND_STORE_URI="postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/mlflow"
MLFLOW_DEFAULT_ARTIFACT_ROOT=./mlruns
```

**FastAPI Settings**

```bash
SECRET_KEY=your-secret-key-here-change-in-production
DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1
```

**Celery Settings**

```bash
CELERY_BROKER_URL="redis://redis:6379/0"
CELERY_RESULT_BACKEND="redis://redis:6379/0"
```

### Optional Environment Variables

**AWS Settings (Production Deployment)**

```bash
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=ap-northeast-2
ECR_REGISTRY=your-account.dkr.ecr.ap-northeast-2.amazonaws.com
```

**SonarQube Settings**

```bash
SONAR_HOST_URL=http://localhost:9000
SONAR_TOKEN=your-sonar-token
```

-----

## 🚀 Quick Start

### Local Development Environment

1.  **Clone the repository**

<!-- end list -->

```bash
git clone https://github.com/209512/mlops-recommendation.git
cd mlops-recommendation-platform
```

2.  **Configure Environment**

<!-- end list -->

```bash
cp .env.example .env
# Edit the necessary values in the .env file
```

3.  **Start Services with Docker**

<!-- end list -->

```bash
docker compose up -d
```

4.  **Access API Documentation**
      - Swagger UI: http://localhost:8000/docs
      - ReDoc: http://localhost:8000/redoc

### Development Server

```bash
# FastAPI Development Server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Celery Worker
celery -A app.infrastructure.celery worker --loglevel=info

# Celery Beat (Scheduler)
celery -A app.infrastructure.celery beat --loglevel=info
```

-----

## 🧪 Testing

```bash
# Run all tests
pytest

# Coverage Report
pytest --cov=app --cov-report=html

# MyPy Type Checking
mypy .

# Ruff Linting
ruff check . --fix
```

-----

## 📊 Monitoring

### Prometheus Metrics

  - http://localhost:8001/metrics

### Grafana Dashboard

  - http://localhost:3000
  - Default Login: admin/admin

### SonarQube

  - http://localhost:9000
  - Project quality analysis and security scanning

-----

## ☁️ Cloud Deployment

### Kubernetes Deployment

```bash
# Create Namespace
kubectl apply -f k8s/namespace.yaml

# ConfigMap and Secret
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# Main Application Deployment
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

### AWS EKS Deployment

```bash
# Configure EKS cluster access
aws eks update-kubeconfig --region <region> --name <cluster-name>

# Push image to ECR
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker build -t mlops-recommendation-platform .
docker tag mlops-recommendation-platform:latest <account-id>.dkr.ecr.<region>.amazonaws.com/mlops-recommendation-platform:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/mlops-recommendation-platform:latest
```

-----

## 🔄 CI/CD Pipeline

### GitHub Actions

  - **CI**: Code quality checks, testing, and building (`.github/workflows/ci.yml`)
  - **CD**: Automated deployment (`.github/workflows/cd.yml`)
  - **Model Retrain**: Scheduled model retraining (`.github/workflows/model-retrain.yml`)

### Code Quality Standards

  - **MyPy**: Strict type checking enforced.
  - **Ruff**: Code formatting and linting.
  - **SonarQube**: Code quality and security analysis.
  - **Test Coverage**: Minimum 80% coverage required.

-----

## 📈 API Usage Examples

### Recommendation Requests

```python
import requests

# User Recommendations
response = requests.get(
    "http://localhost:8000/api/v1/recommendations/users/{user_id}",
    params={"limit": 10}
)

# Lecture Recommendations (Similar Items)
response = requests.get(
    "http://localhost:8000/api/v1/recommendations/lectures/{lecture_id}",
    params={"limit": 5}
)
```

### Model Training

```python
# Model Training Request
response = requests.post(
    "http://localhost:8000/api/v1/training/train",
    json={"algorithm": "als", "factors": 50}
)
```

-----

## 🤝 Contribution Guide

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes (`git commit -m 'Add amazing feature'`).
4.  Push to the branch (`git push origin feature/amazing-feature`).
5.  Open a **Pull Request**.

### Code Quality Standards

  - **MyPy**: Strict type checking is mandatory.
  - **Ruff**: Code formatting and linting.
  - **SonarQube**: Code quality and security analysis.
  - **Test Coverage**: Minimum 80% coverage required.

-----

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

-----

## 🔗 References

  - [Implicit Library Documentation](https://benfred.github.io/implicit/)
  - [FastAPI Documentation](https://fastapi.tiangolo.com/)
  - [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
  - [Kubeflow Documentation](https://www.kubeflow.org/docs/)
  - [Prometheus Documentation](https://prometheus.io/docs/)
  - [Kubernetes Documentation](https://kubernetes.io/docs/)