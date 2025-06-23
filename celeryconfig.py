"""
Celery Beat schedule configuration.

This module defines the periodic tasks that Celery Beat will execute.
Tasks are scheduled using crontab expressions for precise timing control.
"""

from celery.schedules import crontab

beat_schedule = {
    "run-every-1-minute": {
        "task": "tasks.run_market_analysis",
        "schedule": crontab(minute="*/1"),  # every 1 minute
    }
}
