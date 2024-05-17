import os
import redis

print(f"Redis server running on: {os.environ['REDIS_URI']}")

client = redis.Redis.from_url(
  os.environ["REDIS_URI"],
  decode_responses=True
)