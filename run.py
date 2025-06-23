import os

# ✅ Monkey patch BEFORE anything else
import eventlet
eventlet.monkey_patch()

# Now do other imports
from app.models import db
from app import create_app, socketio
import logging

print("start")
app = create_app()
print("end")

with app.app_context():
    db.create_all()
    print("Database tables created (if they didn't exist).")

def main():
    """
    Main entry point for the Flask application.
    
    This function initializes the Flask app, creates database tables,
    and starts the SocketIO server on the specified port.
    """
    logging.basicConfig(level=logging.DEBUG)
    # Use PORT environment variable (Fly.io provides this)
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
