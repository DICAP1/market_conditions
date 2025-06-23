from flask import Blueprint, request, jsonify
import redis, json
from app.models import db, Wishlist


api = Blueprint("api", __name__)
r = redis.Redis()

@api.route("/health")
def health_check():
    """
    Simple health check endpoint for deployment monitoring.
    
    Returns:
        JSON: Status message indicating the service is running
    """
    return jsonify({"status": "healthy", "service": "market-conditions-api"})

@api.route("/api/analysis")
def get_analysis():
    """
    Get market analysis for a specific instrument.
    
    This endpoint retrieves cached market analysis data from Redis
    for the specified instrument. The analysis includes predicted price,
    technical indicators, market signals, and trading recommendations.
    
    Query Parameters:
        instrument (str): The trading instrument symbol (e.g., 'EUR_USD')
        
    Returns:
        JSON: Analysis data including predicted price, indicators, and strategy
              recommendations, or empty object if no analysis found
              
    Status Codes:
        200: Analysis data retrieved successfully
        400: Missing instrument parameter
    """
    instrument = request.args.get("instrument")
    if not instrument:
        return jsonify({"error": "instrument is required"}), 400

    key = f"analysis:{instrument}"
    result = r.get(key)

    if result:
        raw = json.loads(result)
        response = {
            "instrument": instrument,
            "analysis": {
                "predicted_price": raw.get("predicted_price"),
                "best_features": raw.get("best_features"),
                "model_metrics": raw.get("model_metrics"),
                "trend": raw.get("trend"),
                "momentum": raw.get("momentum"),
                "macd": raw.get("macd"),
                "breakout": raw.get("breakout"),
                "volatility": raw.get("volatility"),
                "spread": raw.get("spread"),
                "pinbar": raw.get("pinbar"),
                "anomaly": raw.get("anomaly"),
                "market_condition": raw.get("market_condition"),
                "recommended_strategy": raw.get("recommended_strategy"),
                "strategy_confidence": raw.get("strategy_confidence"),
                "final_strategy": raw.get("final_strategy"),
            }
        }
        return jsonify(response)
    else:
        return jsonify({"instrument": instrument, "analysis": {}})



@api.route("/api/watchlist", methods=["POST"])
def save_watchlist():
    """
    Save a user's watchlist of trading instruments.
    
    This endpoint allows users to save their preferred trading instruments
    to a watchlist. It first clears any existing watchlist for the user
    and then saves the new list of instruments.
    
    Request Body:
        JSON object containing:
            user_id (int): Unique identifier for the user
            instruments (list): List of instrument symbols to save
            
    Returns:
        JSON: Status message indicating success or failure
        
    Status Codes:
        200: Watchlist saved successfully
        400: Missing user_id or instruments parameters
    """
    user_id = request.json.get("user_id")
    instruments = request.json.get("instruments")
    if not user_id or not instruments:
        return jsonify({"status": "error", "message": "Missing user_id or instruments"}), 400

        # Optional: clear previous watchlist
    Wishlist.query.filter_by(user_id=user_id).delete()

    # Insert new watchlist items
    for instrument in instruments:
        item = Wishlist(user_id=user_id, instrument=instrument)
        db.session.add(item)

    db.session.commit()
    return jsonify({"status": "ok", "message": "Watchlist saved"})


@api.route("/api/watchlist", methods=["GET"])
def get_watchlist():
    """
    Retrieve a user's watchlist of trading instruments.
    
    This endpoint fetches the saved watchlist for a specific user
    from the database and returns the list of instrument symbols.
    
    Query Parameters:
        user_id (int): Unique identifier for the user
        
    Returns:
        JSON: Object containing user_id and list of instruments,
              or empty object if no watchlist found
              
    Status Codes:
        200: Watchlist retrieved successfully
        400: Missing user_id parameter
    """
    user_id = request.args.get("user_id")
    print("userasdas", user_id)
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    wishlist_items = Wishlist.query.filter_by(user_id=user_id).all()
    instruments = [item.instrument for item in wishlist_items]
    if wishlist_items:
        return jsonify({
            "user_id": user_id,
            "instruments": instruments
        })
    else:
        return jsonify({})

