"""
Celery Beat scheduler configuration and initialization.

This module creates and configures a Celery Beat instance for scheduling periodic tasks.
The Beat scheduler will execute tasks according to the schedule defined in celeryconfig.py.
This file is used as a module name for Beat CLI to load the Celery app.
"""

from app import create_app, make_celery

# Create Flask app and Celery instance
app = create_app()
celery = make_celery(app)

# You do NOT need to run celery.start() here.
# This file is only used as a module name for Beat CLI to load the Celery app.
