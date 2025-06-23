# Fly.io Deployment Guide

This guide will help you deploy the Market Conditions Analysis application to Fly.io.

## Prerequisites

1. Install Fly CLI: https://fly.io/docs/hands-on/install-flyctl/
2. Sign up for Fly.io account
3. Install Docker (for local testing)

## Pre-deployment Setup

### 1. Set Environment Variables

You'll need to set the following environment variables in Fly.io:

```bash
# Set Flask secret key (generate a secure one)
fly secrets set FLASK_SECRET_KEY="your-secure-secret-key-here"

# Set Redis URL (Fly.io will provide this when you create a Redis app)
fly secrets set REDIS_URL="redis://your-redis-app.internal:6379/0"

# Set database URL (Fly.io will provide this when you create a Postgres app)
fly secrets set DATABASE_URL="postgresql://username:password@host:port/database"
```

### 2. Create Required Services

#### Create PostgreSQL Database
```bash
fly postgres create market-condition-db
fly postgres attach market-condition-db --app market-condition
```

#### Create Redis Instance
```bash
fly redis create market-condition-redis
fly redis attach market-condition-redis --app market-condition
```

## Deployment Steps

### 1. Deploy the Application
```bash
# Deploy to Fly.io
fly deploy

# Check the deployment status
fly status
```

### 2. Scale the Application
```bash
# Scale to 1 instance (free tier)
fly scale count 1

# Scale to multiple instances (paid tier)
fly scale count 3
```

### 3. Monitor the Application
```bash
# View logs
fly logs

# Open the application
fly open

# Check app status
fly status
```

## Configuration Files

### fly.toml
- Configured for Python Flask application
- Includes health checks
- Set up for HTTP/HTTPS traffic
- Configured for immediate deployment strategy

### Dockerfile
- Uses Python 3.12 slim image
- Installs TA-Lib system dependencies
- Sets up non-root user for security
- Configures proper environment variables

### Environment Variables Required
- `FLASK_SECRET_KEY`: Secure secret key for Flask sessions
- `REDIS_URL`: Redis connection string
- `DATABASE_URL`: PostgreSQL connection string
- `PORT`: Port number (automatically set by Fly.io)

## Health Checks

The application includes health checks that:
- Check the `/api/analysis` endpoint every 30 seconds
- Verify the application is responding to HTTP requests
- Restart the application if health checks fail

## Troubleshooting

### Common Issues

1. **TA-Lib Installation Fails**
   - The Dockerfile includes TA-Lib system dependencies
   - If issues persist, check the build logs: `fly logs`

2. **Database Connection Issues**
   - Ensure DATABASE_URL is properly set
   - Check if PostgreSQL app is running: `fly postgres list`

3. **Redis Connection Issues**
   - Ensure REDIS_URL is properly set
   - Check if Redis app is running: `fly redis list`

4. **Application Won't Start**
   - Check logs: `fly logs`
   - Verify environment variables: `fly secrets list`
   - Check app status: `fly status`

### Useful Commands

```bash
# View detailed logs
fly logs --all

# SSH into the running container
fly ssh console

# Check environment variables
fly ssh console -C "env | grep -E '(FLASK|REDIS|DATABASE)'"

# Restart the application
fly apps restart market-condition

# Scale down to 0 (stop the app)
fly scale count 0

# Scale up to 1 (start the app)
fly scale count 1
```

## Performance Optimization

1. **Database Connection Pooling**: The application uses SQLAlchemy with connection pooling
2. **Redis Caching**: Analysis results are cached in Redis
3. **Background Tasks**: Celery handles background market analysis tasks
4. **Health Checks**: Automatic health monitoring and restart

## Security Considerations

1. **Non-root User**: Application runs as non-root user in container
2. **Environment Variables**: Sensitive data stored in environment variables
3. **HTTPS**: Automatic HTTPS termination by Fly.io
4. **Secret Management**: Use Fly.io secrets for sensitive configuration

## Monitoring and Logging

- Application logs are available via `fly logs`
- Health check status is monitored automatically
- Database and Redis metrics available in Fly.io dashboard
- Set up alerts for application failures

## Cost Optimization

- Use free tier for development/testing
- Scale down to 0 when not in use
- Monitor resource usage in Fly.io dashboard
- Use appropriate instance sizes for your workload 