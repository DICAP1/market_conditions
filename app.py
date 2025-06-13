# Always call this first before importing any other modules
from eventlet import monkey_patch
monkey_patch()
print("Flask app is starting...")  # Add this print statement
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import redis
import os
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# Connect to Redis (already hosted on Fly.io)
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
r = redis.from_url(redis_url)

# Set up SocketIO with Redis as the message queue
socketio = SocketIO(app, cors_allowed_origins="*", message_queue=redis_url)

# --- REST Endpoint ---
@app.route('/api/analysis')
def get_analysis():
    instrument = request.args.get('instrument')
    if not instrument:
        return jsonify({"error": "Missing instrument param"}), 400

    data = r.get(f"analysis:{instrument}")
    if not data:
        return jsonify({"error": "No data found"}), 404

    return jsonify(json.loads(data))

# --- REST Endpoint to save user watchlist (optional) ---
@app.route('/api/watchlist', methods=['POST'])
def save_watchlist():
    data = request.get_json()
    r.set("user:watchlist", json.dumps(data.get("watchlist", [])))
    return jsonify({"message": "Watchlist saved"})

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    data = r.get("user:watchlist")
    return jsonify(json.loads(data)) if data else jsonify([])

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
