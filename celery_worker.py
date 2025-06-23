"""
Celery worker configuration and initialization.

This module creates and configures a Celery worker instance with Flask app context support.
The worker can execute background tasks with access to Flask's application context.
"""

from app import create_app, make_celery

app = create_app()
celery = make_celery(app)
