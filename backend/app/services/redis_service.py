"""
Redis service for caching and real-time data management.
Handles connection management, data caching, and pub/sub messaging.
"""

import json
import asyncio
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta

import redis.asyncio as redis
import structlog

from app.core.config import settings

logger = structlog.get_logger()


class RedisService:
    """
    Async Redis service for caching and real-time data operations.
    Provides high-level interface for common Redis operations.
    """
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        
    async def connect(self):
        """Establish connection to Redis server"""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            
            logger.info("Redis connection established", 
                       url=settings.REDIS_URL.split('@')[-1])  # Hide credentials
                       
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise
    
    async def disconnect(self):
        """Close Redis connection"""
        try:
            if self.pubsub:
                await self.pubsub.close()
            
            if self.redis_client:
                await self.redis_client.close()
                
            logger.info("Redis connection closed")
            
        except Exception as e:
            logger.error("Error closing Redis connection", error=str(e))
    
    async def ping(self) -> bool:
        """Test Redis connection"""
        try:
            if self.redis_client:
                await self.redis_client.ping()
                return True
            return False
        except Exception:
            return False
    
    # Basic Operations
    
    async def set(self, key: str, value: Union[str, Dict, List], expire: Optional[int] = None):
        """Set a key-value pair with optional expiration"""
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            if expire:
                await self.redis_client.setex(key, expire, value)
            else:
                await self.redis_client.set(key, value)
                
        except Exception as e:
            logger.error("Failed to set Redis key", key=key, error=str(e))
            raise
    
    async def get(self, key: str, parse_json: bool = False) -> Optional[Union[str, Dict, List]]:
        """Get value by key with optional JSON parsing"""
        try:
            value = await self.redis_client.get(key)
            
            if value is None:
                return None
            
            if parse_json:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            
            return value
            
        except Exception as e:
            logger.error("Failed to get Redis key", key=key, error=str(e))
            return None
    
    async def delete(self, *keys: str) -> int:
        """Delete one or more keys"""
        try:
            return await self.redis_client.delete(*keys)
        except Exception as e:
            logger.error("Failed to delete Redis keys", keys=keys, error=str(e))
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return bool(await self.redis_client.exists(key))
        except Exception as e:
            logger.error("Failed to check Redis key existence", key=key, error=str(e))
            return False
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration time for a key"""
        try:
            return bool(await self.redis_client.expire(key, seconds))
        except Exception as e:
            logger.error("Failed to set Redis key expiration", key=key, error=str(e))
            return False
    
    # Hash Operations
    
    async def hset(self, name: str, mapping: Dict[str, Any]):
        """Set multiple fields in a hash"""
        try:
            # Convert values to strings, JSON encode complex types
            processed_mapping = {}
            for key, value in mapping.items():
                if isinstance(value, (dict, list)):
                    processed_mapping[key] = json.dumps(value)
                else:
                    processed_mapping[key] = str(value)
            
            await self.redis_client.hset(name, mapping=processed_mapping)
            
        except Exception as e:
            logger.error("Failed to set Redis hash", name=name, error=str(e))
            raise
    
    async def hget(self, name: str, key: str, parse_json: bool = False) -> Optional[Union[str, Dict, List]]:
        """Get a field from a hash"""
        try:
            value = await self.redis_client.hget(name, key)
            
            if value is None:
                return None
            
            if parse_json:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            
            return value
            
        except Exception as e:
            logger.error("Failed to get Redis hash field", name=name, key=key, error=str(e))
            return None
    
    async def hgetall(self, name: str, parse_json_values: bool = False) -> Dict[str, Any]:
        """Get all fields from a hash"""
        try:
            data = await self.redis_client.hgetall(name)
            
            if parse_json_values:
                parsed_data = {}
                for key, value in data.items():
                    try:
                        parsed_data[key] = json.loads(value)
                    except json.JSONDecodeError:
                        parsed_data[key] = value
                return parsed_data
            
            return data
            
        except Exception as e:
            logger.error("Failed to get Redis hash", name=name, error=str(e))
            return {}
    
    # List Operations
    
    async def lpush(self, key: str, *values: Union[str, Dict, List]):
        """Push values to the left of a list"""
        try:
            processed_values = []
            for value in values:
                if isinstance(value, (dict, list)):
                    processed_values.append(json.dumps(value))
                else:
                    processed_values.append(str(value))
            
            await self.redis_client.lpush(key, *processed_values)
            
        except Exception as e:
            logger.error("Failed to push to Redis list", key=key, error=str(e))
            raise
    
    async def rpush(self, key: str, *values: Union[str, Dict, List]):
        """Push values to the right of a list"""
        try:
            processed_values = []
            for value in values:
                if isinstance(value, (dict, list)):
                    processed_values.append(json.dumps(value))
                else:
                    processed_values.append(str(value))
            
            await self.redis_client.rpush(key, *processed_values)
            
        except Exception as e:
            logger.error("Failed to push to Redis list", key=key, error=str(e))
            raise
    
    async def lrange(self, key: str, start: int = 0, end: int = -1, parse_json: bool = False) -> List[Any]:
        """Get a range of elements from a list"""
        try:
            values = await self.redis_client.lrange(key, start, end)
            
            if parse_json:
                parsed_values = []
                for value in values:
                    try:
                        parsed_values.append(json.loads(value))
                    except json.JSONDecodeError:
                        parsed_values.append(value)
                return parsed_values
            
            return values
            
        except Exception as e:
            logger.error("Failed to get Redis list range", key=key, error=str(e))
            return []
    
    async def ltrim(self, key: str, start: int, end: int):
        """Trim a list to the specified range"""
        try:
            await self.redis_client.ltrim(key, start, end)
        except Exception as e:
            logger.error("Failed to trim Redis list", key=key, error=str(e))
    
    # Pub/Sub Operations
    
    async def publish(self, channel: str, message: Union[str, Dict, List]):
        """Publish a message to a channel"""
        try:
            if isinstance(message, (dict, list)):
                message = json.dumps(message)
            
            await self.redis_client.publish(channel, message)
            
        except Exception as e:
            logger.error("Failed to publish Redis message", channel=channel, error=str(e))
            raise
    
    async def subscribe(self, *channels: str):
        """Subscribe to channels"""
        try:
            if not self.pubsub:
                self.pubsub = self.redis_client.pubsub()
            
            await self.pubsub.subscribe(*channels)
            
        except Exception as e:
            logger.error("Failed to subscribe to Redis channels", channels=channels, error=str(e))
            raise
    
    async def get_message(self, timeout: Optional[float] = None, parse_json: bool = False) -> Optional[Dict[str, Any]]:
        """Get a message from subscribed channels"""
        try:
            if not self.pubsub:
                return None
            
            message = await self.pubsub.get_message(timeout=timeout)
            
            if message and message['type'] == 'message':
                data = message['data']
                if parse_json and isinstance(data, str):
                    try:
                        message['data'] = json.loads(data)
                    except json.JSONDecodeError:
                        pass
                
                return message
            
            return None
            
        except Exception as e:
            logger.error("Failed to get Redis message", error=str(e))
            return None
    
    # Cache Management
    
    async def cache_telemetry_data(self, session_id: str, data: Dict[str, Any], max_points: int = 10000):
        """Cache telemetry data with automatic trimming"""
        try:
            key = f"telemetry:{session_id}"
            
            # Add timestamp if not present
            if 'timestamp' not in data:
                data['timestamp'] = datetime.utcnow().isoformat()
            
            # Push to list
            await self.lpush(key, data)
            
            # Trim to max points to prevent unlimited growth
            await self.ltrim(key, 0, max_points - 1)
            
            # Set expiration (24 hours)
            await self.expire(key, 86400)
            
        except Exception as e:
            logger.error("Failed to cache telemetry data", session_id=session_id, error=str(e))
    
    async def get_cached_telemetry(self, session_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get cached telemetry data"""
        try:
            key = f"telemetry:{session_id}"
            return await self.lrange(key, 0, limit - 1, parse_json=True)
        except Exception as e:
            logger.error("Failed to get cached telemetry", session_id=session_id, error=str(e))
            return []
    
    async def cache_session_analytics(self, session_id: str, analytics: Dict[str, Any], expire_hours: int = 24):
        """Cache session analytics"""
        try:
            key = f"analytics:{session_id}"
            await self.set(key, analytics, expire=expire_hours * 3600)
        except Exception as e:
            logger.error("Failed to cache session analytics", session_id=session_id, error=str(e))
    
    async def get_cached_analytics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached session analytics"""
        try:
            key = f"analytics:{session_id}"
            return await self.get(key, parse_json=True)
        except Exception as e:
            logger.error("Failed to get cached analytics", session_id=session_id, error=str(e))
            return None
    
    # Utility Methods
    
    async def cleanup_expired_keys(self):
        """Clean up expired telemetry and cache keys"""
        try:
            # This is a basic cleanup - in production you'd want more sophisticated cleanup
            pattern_keys = [
                "telemetry:*",
                "analytics:*", 
                "session:*"
            ]
            
            for pattern in pattern_keys:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    # Check TTL and clean up keys without expiration
                    for key in keys:
                        ttl = await self.redis_client.ttl(key)
                        if ttl == -1:  # No expiration set
                            await self.expire(key, 86400)  # Set 24 hour expiration
            
            logger.info("Redis cleanup completed")
            
        except Exception as e:
            logger.error("Failed to cleanup Redis keys", error=str(e))
