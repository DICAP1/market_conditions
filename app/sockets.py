from flask_socketio import SocketIO, emit

# Already initialized somewhere like this:
socketio = SocketIO(cors_allowed_origins="*", message_queue="redis://localhost:6379/0")  # or your Redis URL

@socketio.on('connect')
def handle_connect():
    """
    Handle WebSocket client connection.
    
    This function is called when a client connects to the WebSocket.
    It sends an acknowledgment message back to the client confirming
    the successful connection.
    """
    print("✅ WebSocket client connected")
    emit("connection_ack", {"message": "WebSocket connection successful"})

@socketio.on('disconnect')
def handle_disconnect():
    """
    Handle WebSocket client disconnection.
    
    This function is called when a client disconnects from the WebSocket.
    It logs the disconnection event for monitoring purposes.
    """
    print("❌ WebSocket client disconnected")

@socketio.on('ping')
def handle_ping():
    """
    Handle WebSocket ping message.
    
    This function responds to ping messages from clients with a pong
    response, which can be used for connection health monitoring
    and keep-alive functionality.
    """
    emit("pong", {"message": "pong!"})
