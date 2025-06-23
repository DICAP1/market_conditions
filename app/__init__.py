# app/__init__.py
from celery.schedules import crontab
from flask import Flask
from flask_socketio import SocketIO
from celery import Celery
import redis
import os
from flask_migrate import Migrate, upgrade
from .models import db
migrate = Migrate()
# --- Configuration ---
# It's good practice to get Redis URL and Secret Key from environment variables
# For local development, you might hardcode them, but use os.environ.get for production.
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "super_secret_key_change_me")

# --- Global Extensions Initialization (unbound for app factory pattern) ---
# Initialize Celery here before the app factory, as tasks might import it directly.
celery = Celery(__name__, broker=REDIS_URL, backend=REDIS_URL)

# Initialize SocketIO without an app instance yet. It will be bound in create_app.
socketio = SocketIO()

# Initialize redis_client for direct Redis operations (e.g., storing analysis data).
# decode_responses=True ensures strings are returned, not bytes.
redis_client = redis.StrictRedis.from_url(REDIS_URL, decode_responses=True)

# --- Application Factory Function ---
def create_app():
    """
    Creates and configures the Flask application.
    
    This function initializes Flask, SocketIO, and configures Celery with the
    Flask application context. It sets up database connections, migrations,
    and registers blueprints.
    
    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__)
    app.config["SECRET_KEY"] = FLASK_SECRET_KEY

    # --- Initialize SocketIO with the Flask app instance ---
    # Crucial step: Bind SocketIO to the Flask app.
    # message_queue: Allows Celery workers to emit messages via Redis.
    # async_mode: Must be "eventlet" because eventlet.monkey_patch() is used.
    # cors_allowed_origins: Set to "*" for development; be specific in production
    # (e.g., "http://yourfrontend.com", "http://localhost:3000").
    
    # Use environment variable for database URL (Fly.io provides DATABASE_URL)
    # database_url = os.environ.get(
    #     "DATABASE_URL",
    #     "postgresql://pgdb-1zqyxr7zz10wp8mk:HwmyvNe6LwueJyWtQuejKy2b@pgbouncer.1zqyxr7zz10wp8mk.flympg.net/pgdb-1zqyxr7zz10wp8mk"
    # )
    database_url = os.environ.get("DATABASE_URL", "postgresql://postgres:admin@localhost/market-condition")
    # Fix for Fly.io PostgreSQL URL format
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)
    from .models import Wishlist
    migrate.init_app(app, db)
    socketio.init_app(app,
                      message_queue=REDIS_URL,
                      async_mode="eventlet",
                      cors_allowed_origins="*")

    with app.app_context():
        upgrade()
    # --- Configure Celery with Flask app context ---
    # This custom Task class ensures Celery tasks can access Flask's app context,
    # which is often needed for database operations, configuration, etc.
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    celery.Task = ContextTask

    # Update Celery's configuration with any settings from the Flask app.
    celery.conf.update(app.config)

    # --- Celery Beat Schedule (for scheduled tasks) ---
    # This dictionary defines tasks to be run by Celery Beat.
    # 'app.tasks.analyze_instruments_task' refers to the name you give your Celery task.
    # crontab(minute='*') schedules it to run every minute.
    celery.conf.beat_schedule = {
        'run-analysis-every-minute': {
            'task': 'app.tasks.analyze_instruments_task',
            'schedule': crontab(minute='*')
        }
    }

    # --- Import and Register Blueprints/Routes ---
    # Import routes AFTER the app is created and configured to avoid circular imports.
    # Assuming you have a 'routes.py' file defining a blueprint named 'api'.
    from . import routes
    app.register_blueprint(routes.api)

    return app

def make_celery(app):
    """
    Creates and configures a Celery instance with Flask app context support.
    
    This function sets up Celery with the Flask application context,
    configures the beat schedule for periodic tasks, and ensures
    Celery tasks can access Flask's app context for database operations.
    
    Args:
        app (Flask): Flask application instance
        
    Returns:
        Celery: Configured Celery instance with Flask context support
    """
    # Update config from Flask app
    celery.conf.update(app.config)

    # Flask context support in Celery tasks
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    celery.Task = ContextTask

    # Celery Beat Schedule
    celery.conf.beat_schedule = {
        'run-analysis-every-minute': {
            'task': 'app.tasks.analyze_instruments_task',  # Must match your actual @celery.task name
            'schedule': crontab(minute='*')
        }
    }

    return celery


# App and Celery are created once at the package level
app = create_app()
make_celery(app)  # Attach Flask context

# ❗ Import tasks after Celery is ready
from . import tasks
