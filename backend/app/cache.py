import os
import aioredis

REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379/0')
redis = None


async def init_redis():
    global redis
    redis = await aioredis.from_url(REDIS_URL)


async def get(key: str):
    if not redis:
        await init_redis()
    return await redis.get(key)


async def set(key: str, value, expire: int = None):
    if not redis:
        await init_redis()
    await redis.set(key, value, ex=expire)
