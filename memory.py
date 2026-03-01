from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config import cfg


@dataclass
class Message:
    role: str           # 'user' or 'assistant'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    context_flight_ids: list[str] = field(default_factory=list)


@dataclass
class SessionMemory:
    session_id: str
    messages: list[Message] = field(default_factory=list)
    last_retrieved_flights: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_user_message(self, content: str):
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str, flight_ids: list[str] = None):
        self.messages.append(Message(
            role="assistant",
            content=content,
            context_flight_ids=flight_ids or [],
        ))

    def get_recent_messages(self, n: int = None) -> list[Message]:
        """Sliding window — last N messages."""
        window = n or cfg.MEMORY_WINDOW
        return self.messages[-window:]

    def get_chat_history(self, n: int = None) -> list[dict]:
        """Returns messages in Groq/OpenAI format for prompt building."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.get_recent_messages(n)
        ]

    def get_context_summary(self) -> str:
        if not self.last_retrieved_flights:
            return ""
        lines = ["Recently discussed flights:"]
        for f in self.last_retrieved_flights[:3]:
            airline = f.get("airline", "")
            code = f.get("flight_code", "")
            dep = f.get("dep_time", "")
            fare = f.get("cheapest_fare", "")
            stops = "Non-stop" if f.get("stops", 0) == 0 else f"{f.get('stops')} stop(s)"
            refund = "Refundable" if f.get("is_refundable") else "Non-refundable"
            lines.append(f"  - {airline} {code}: {dep} | ₹{fare} | {stops} | {refund}")
        return "\n".join(lines)

    def update_flight_context(self, flights: list[dict]):
        self.last_retrieved_flights = flights

    def clear(self):
        self.messages = []
        self.last_retrieved_flights = []

    @property
    def message_count(self) -> int:
        return len(self.messages)


class MemoryManager:

    def __init__(self):
        self._sessions: dict[str, SessionMemory] = {}

    def get_or_create(self, session_id: str) -> SessionMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionMemory(session_id=session_id)
            print(f"[memory] 🆕 New session: {session_id}")
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[SessionMemory]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def all_session_ids(self) -> list[str]:
        return list(self._sessions.keys())

    def count(self) -> int:
        return len(self._sessions)

memory_manager = MemoryManager()
