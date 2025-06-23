"""
Celery configuration for the Flask application.

This module defines Celery broker settings and beat schedule configuration
for background task processing in the market analysis application.
"""

from celery.schedules import crontab

CELERY_BROKER_URL = "redis://localhost:6379/0"
CELERY_RESULT_BACKEND = "redis://localhost:6379/0"

CELERY_BEAT_SCHEDULE = {
    'run-analysis-every-minute': {
        'task': 'app.tasks.run_market_analysis',
        'schedule': crontab(minute='*/1'),
    },
}
