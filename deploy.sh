#!/bin/bash

# Hajj Financial Sustainability Application - Production Deployment Script

set -e

echo "�� Starting deployment of Hajj Financial Sustainability Application..."

# Check if required environment variables are set
if [ -z "$DEPLOYMENT_ENV" ]; then
    echo "❌ DEPLOYMENT_ENV not set. Please set it to 'staging' or 'production'"
    exit 1
fi

echo "� Deployment Environment: $DEPLOYMENT_ENV"

# Build and deploy with Docker
echo "� Building Docker image..."
docker build -t hajj-finance-app:latest .

# Tag for deployment
if [ "$DEPLOYMENT_ENV" = "production" ]; then
    docker tag hajj-finance-app:latest hajj-finance-app:prod
    echo "��️ Tagged for production deployment"
elif [ "$DEPLOYMENT_ENV" = "staging" ]; then
    docker tag hajj-finance-app:latest hajj-finance-app:staging
    echo "�️ Tagged for staging deployment"
fi

# Run health checks
echo "� Running health checks..."
docker run --rm -d --name hajj-app-test -p 8502:8501 hajj-finance-app:latest

# Wait for application to start
sleep 30

# Test health endpoint
if curl -f http://localhost:8502/_stcore/health; then
    echo "✅ Health check passed"
    docker stop hajj-app-test
else
    echo "❌ Health check failed"
    docker stop hajj-app-test
    exit 1
fi

# Deploy based on environment
if [ "$DEPLOYMENT_ENV" = "production" ]; then
    echo "� Deploying to production..."
    # Add production deployment commands here
    # e.g., push to container registry, update kubernetes deployment, etc.
    
elif [ "$DEPLOYMENT_ENV" = "staging" ]; then
    echo "� Deploying to staging..."
    # Add staging deployment commands here
fi

echo "� Deployment completed successfully!"
