import redis
from typing import List, Dict
import json

class SessionManager:
    def __init__(self, host: str = 'redis', port: int = 6379, password: str = None):
        self.redis = redis.Redis(host=host, port=port, password=password, decode_responses=True)
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to conversation history"""
        message = {'role': role, 'content': content}
        self.redis.rpush(f"session:{session_id}:history", json.dumps(message))
    
    def get_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        history = self.redis.lrange(f"session:{session_id}:history", 0, -1)
        return [json.loads(msg) for msg in history]
    
    def clear_history(self, session_id: str):
        """Clear conversation history"""
        self.redis.delete(f"session:{session_id}:history")
