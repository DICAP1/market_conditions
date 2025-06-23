from app import celery, socketio, redis_client
from analysis.market_conditions_analyser4 import analyze_instruments

@celery.task(name="app.tasks.analyze_instruments_task")
def analyze_instruments_task():
    """
    Celery task for running market analysis on all instruments.
    
    This task is scheduled to run periodically (every minute) and
    performs comprehensive market analysis on all configured trading
    instruments. It fetches real-time data, computes technical indicators,
    generates predictions, and stores results in Redis for API access.
    
    The task also emits real-time updates via SocketIO if the socketio
    instance is available.
    
    Returns:
        dict: Analysis results for all instruments
    """
    print("🔁 Running market analysis task")
    analyze_instruments(socketio)
