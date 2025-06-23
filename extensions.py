# extensions.py
import redis

# Change host/port if you're using a remote Redis
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
