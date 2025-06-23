from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Wishlist(db.Model):
    """
    Database model for storing user watchlists.
    
    This model represents a user's watchlist of trading instruments.
    Each record contains a user ID and an instrument symbol that
    the user wants to track.
    
    Attributes:
        id (int): Primary key, auto-incrementing unique identifier
        user_id (int): Foreign key to identify the user (not nullable)
        instrument (str): Trading instrument symbol (e.g., 'EUR_USD') (not nullable)
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    instrument = db.Column(db.String(255), nullable=False)
