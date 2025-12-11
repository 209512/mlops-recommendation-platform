#!/bin/bash
set -e

# MLflow 데이터베이스 생성 (이미 존재하면 무시)
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE mlflow'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO $POSTGRES_USER;
EOSQL

echo "MLflow database created successfully"

# MLflow 스키마 마이그레이션
echo "Running MLflow database schema migration..."
mlflow db upgrade postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@db:5432/mlflow

echo "MLflow database initialization completed"