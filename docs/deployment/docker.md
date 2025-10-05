# Docker Deployment Guide

This guide covers deploying the RAG Agent System using Docker containers for development, testing, and production environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Development Environment](#development-environment)
- [Production Environment](#production-environment)
- [Docker Compose Setup](#docker-compose-setup)
- [Configuration](#configuration)
- [Scaling](#scaling)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Docker 20.10+ installed
- Docker Compose 2.0+ installed
- 4GB+ available RAM
- API keys for LLM providers (OpenAI, etc.)

### Check Prerequisites

```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker compose version

# Verify Docker is running
docker info
```

## Quick Start

### 1. Build the Base Image

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt requirements_langgraph.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_langgraph.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY dashboards/ ./dashboards/

# Create necessary directories
RUN mkdir -p data/raw data/processed logs

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from src.core.rag_system import RAGSystem; rag = RAGSystem(); print('OK')" || exit 1

# Default command
CMD ["python", "-m", "src.core.rag_system"]
```

### 2. Build and Run

```bash
# Build the image
docker build -t rag-agent-system .

# Run with environment variables
docker run -d \
  --name rag-system \
  -p 8000:8000 \
  -p 8501:8501 \
  -e OPENAI_API_KEY=your-api-key \
  -e RAG_ENVIRONMENT=development \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  rag-agent-system
```

### 3. Access the System

```bash
# Check container status
docker ps

# View logs
docker logs rag-system

# Access the dashboard
open http://localhost:8501
```

## Development Environment

### Development Dockerfile

```dockerfile
# Dockerfile.dev
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including dev tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements_dev.txt requirements_langgraph.txt ./

# Install all dependencies including dev dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_dev.txt && \
    pip install --no-cache-dir -r requirements_langgraph.txt

# Copy source code (will be overridden by volume in dev)
COPY . .

# Create directories
RUN mkdir -p data/raw data/processed logs

# Development ports
EXPOSE 8000 8501 8888

# Install Jupyter for development
RUN pip install jupyter

# Development command with auto-reload
CMD ["python", "-m", "src.core.rag_system", "--reload"]
```

### Build Development Image

```bash
# Build development image
docker build -f Dockerfile.dev -t rag-agent-system:dev .

# Run development container with volume mounting
docker run -d \
  --name rag-system-dev \
  -p 8000:8000 \
  -p 8501:8501 \
  -p 8888:8888 \
  -e OPENAI_API_KEY=your-api-key \
  -e RAG_ENVIRONMENT=development \
  -v $(pwd):/app \
  -v $(pwd)/data:/app/data \
  rag-agent-system:dev
```

### Development with Hot Reload

```bash
# For file watching and auto-reload
docker run -d \
  --name rag-system-dev \
  -p 8000:8000 \
  -p 8501:8501 \
  -e OPENAI_API_KEY=your-api-key \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/dashboards:/app/dashboards \
  rag-agent-system:dev \
  watchmedo auto-restart --directory=./src --pattern=*.py --recursive -- python -m src.core.rag_system
```

## Production Environment

### Production Dockerfile

```dockerfile
# Dockerfile.prod
FROM python:3.9-slim as builder

# Build stage
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements_langgraph.txt ./

# Install dependencies
RUN pip install --user --no-cache-dir -r requirements.txt && \
    pip install --user --no-cache-dir -r requirements_langgraph.txt

# Production stage
FROM python:3.9-slim

# Create non-root user
RUN groupadd -r raguser && useradd -r -g raguser raguser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/raguser/.local

# Update PATH
ENV PATH="/home/raguser/.local/bin:${PATH}"

# Copy application code
COPY --chown=raguser:raguser src/ ./src/
COPY --chown=raguser:raguser config/ ./config/
COPY --chown=raguser:raguser dashboards/ ./dashboards/

# Create necessary directories
RUN mkdir -p data/raw data/processed logs && \
    chown -R raguser:raguser data logs

# Switch to non-root user
USER raguser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from src.core.rag_system import RAGSystem; rag = RAGSystem(); print('OK')" || exit 1

# Production command
CMD ["python", "-m", "src.core.rag_system", "--host", "0.0.0.0", "--port", "8000"]
```

### Build Production Image

```bash
# Build production image
docker build -f Dockerfile.prod -t rag-agent-system:prod .

# Run production container
docker run -d \
  --name rag-system-prod \
  --restart unless-stopped \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your-api-key \
  -e RAG_ENVIRONMENT=production \
  -v rag-data:/app/data \
  -v rag-logs:/app/logs \
  --memory="4g" \
  --cpus="2" \
  rag-agent-system:prod
```

## Docker Compose Setup

### Basic Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-system:
    build: .
    container_name: rag-system
    ports:
      - "8000:8000"
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - RAG_ENVIRONMENT=development
      - VECTOR_DB_HOST=chroma
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - chroma
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "from src.core.rag_system import RAGSystem; rag = RAGSystem(); print('OK')"]
      interval: 30s
      timeout: 10s
      retries: 3

  chroma:
    image: chromadb/chroma:latest
    container_name: rag-chroma
    ports:
      - "8080:8000"
    volumes:
      - chroma-data:/chroma/chroma
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: rag-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  chroma-data:
  redis-data:

networks:
  default:
    name: rag-network
```

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  rag-system:
    build:
      context: .
      dockerfile: Dockerfile.prod
    container_name: rag-system-prod
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - RAG_ENVIRONMENT=production
      - VECTOR_DB_HOST=chroma
      - REDIS_HOST=redis
    volumes:
      - rag-data:/app/data
      - rag-logs:/app/logs
    depends_on:
      chroma:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'

  chroma:
    image: chromadb/chroma:latest
    container_name: rag-chroma-prod
    environment:
      - CHROMA_SERVER_AUTHN_CREDENTIALS_FILE=/chroma/auth.txt
    volumes:
      - chroma-data:/chroma/chroma
      - ./config/chroma-auth.txt:/chroma/auth.txt:ro
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1'

  redis:
    image: redis:7-alpine
    container_name: rag-redis-prod
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis-data:/data
      - ./config/redis.conf:/etc/redis/redis.conf:ro
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

  nginx:
    image: nginx:alpine
    container_name: rag-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - rag-system
    restart: unless-stopped

volumes:
  rag-data:
    driver: local
  rag-logs:
    driver: local
  chroma-data:
    driver: local
  redis-data:
    driver: local

networks:
  default:
    name: rag-production
    driver: bridge
```

### Run with Docker Compose

```bash
# Development
docker compose up -d

# Production
docker compose -f docker-compose.prod.yml up -d

# Scale services
docker compose up -d --scale rag-system=3

# View logs
docker compose logs -f rag-system

# Stop services
docker compose down

# Stop and remove volumes
docker compose down -v
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# .env
OPENAI_API_KEY=your-openai-api-key
RAG_ENVIRONMENT=production
RAG_LOG_LEVEL=INFO

# Vector Database
VECTOR_DB_PROVIDER=chroma
VECTOR_DB_HOST=chroma
VECTOR_DB_PORT=8000

# Redis Cache
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password

# Application Settings
RAG_MAX_WORKERS=4
RAG_TIMEOUT_SECONDS=300

# Security
RAG_SECRET_KEY=your-secret-key
RAG_ALLOWED_HOSTS=localhost,127.0.0.1
```

### Volume Mounts

```bash
# Create persistent volumes
docker volume create rag-data
docker volume create rag-logs
docker volume create chroma-data

# Mount specific directories
docker run -d \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  rag-agent-system
```

### Configuration Files

Mount configuration files:

```yaml
services:
  rag-system:
    volumes:
      - ./config/production.yaml:/app/config/config.yaml:ro
      - ./config/logging.yaml:/app/config/logging.yaml:ro
```

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  rag-system:
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    
  nginx:
    image: nginx:alpine
    volumes:
      - ./config/nginx-loadbalancer.conf:/etc/nginx/nginx.conf:ro
    ports:
      - "80:80"
    depends_on:
      - rag-system
```

### Load Balancer Configuration

```nginx
# config/nginx-loadbalancer.conf
upstream rag_backend {
    server rag-system-1:8000;
    server rag-system-2:8000;
    server rag-system-3:8000;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://rag_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### Docker Swarm Deployment

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.swarm.yml rag-stack

# Scale service
docker service scale rag-stack_rag-system=5

# View services
docker service ls

# Update service
docker service update --image rag-agent-system:v2.0 rag-stack_rag-system
```

## Monitoring

### Health Checks

```dockerfile
# Custom health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

### Logging

```yaml
services:
  rag-system:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "10"
```

### Monitoring Stack

```yaml
# monitoring/docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana

  cadvisor:
    image: gcr.io/cadvisor/cadvisor
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro

volumes:
  grafana-data:
```

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker logs rag-system

# Check resource usage
docker stats rag-system

# Inspect container
docker inspect rag-system
```

#### Memory Issues

```bash
# Set memory limits
docker run -d --memory="4g" --memory-swap="6g" rag-agent-system

# Monitor memory usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

#### Network Issues

```bash
# Check network connectivity
docker network ls
docker network inspect rag-network

# Test connectivity between containers
docker exec rag-system ping chroma
```

#### Permission Issues

```bash
# Fix file permissions
docker exec -u root rag-system chown -R raguser:raguser /app/data

# Run with correct user
docker run -d --user $(id -u):$(id -g) rag-agent-system
```

### Debugging Commands

```bash
# Interactive shell
docker exec -it rag-system /bin/bash

# View container processes
docker exec rag-system ps aux

# Check disk usage
docker exec rag-system df -h

# View environment variables
docker exec rag-system env

# Test API endpoints
docker exec rag-system curl -f http://localhost:8000/health
```

### Performance Tuning

#### Resource Allocation

```yaml
services:
  rag-system:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

#### Optimize Image Size

```dockerfile
# Multi-stage build to reduce image size
FROM python:3.9-slim as builder
# ... build dependencies

FROM python:3.9-slim
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
# ... rest of production image
```

### Security Best Practices

1. **Use non-root user**
2. **Keep images updated**
3. **Scan for vulnerabilities**
4. **Limit container capabilities**
5. **Use secrets management**

```bash
# Scan image for vulnerabilities
docker scout cves rag-agent-system:latest

# Run security benchmark
docker run --rm -it \
  -v /var/run/docker.sock:/var/run/docker.sock \
  docker/docker-bench-security
```

This comprehensive Docker deployment guide provides everything needed to containerize and deploy the RAG Agent System in various environments, from development to production scale.