"""Session memory: logs searches and candidate views, persisted to data/memory/."""
import json
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.chat_history import InMemoryChatMessageHistory

from utils.config import MEMORY_DIR
from utils.utils import get_logger

logger = get_logger(__name__)
MEMORY_FILE = MEMORY_DIR / "interactions.json"


class MemoryRAG:
    def __init__(self, persist: bool = True):
        self.persist = persist
        self.chat_history = InMemoryChatMessageHistory()  # LangChain-compatible message log
        self.events: List[Dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------ persistence
    def _load(self) -> None:
        if self.persist and MEMORY_FILE.exists():
            try:
                self.events = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
                for event in self.events:
                    self._mirror(event)
            except Exception as e:
                logger.warning(f"Could not load memory: {e}")
                self.events = []

    def _save(self) -> None:
        if self.persist:
            MEMORY_FILE.write_text(json.dumps(self.events, indent=2), encoding="utf-8")

    def _mirror(self, event: Dict[str, Any]) -> None:
        """Copy an event into the LangChain chat history as a human/AI message pair."""
        if event["type"] == "search":
            self.chat_history.add_user_message(f"Search: {event['query']}")
            self.chat_history.add_ai_message(f"Found {event['results_count']} results")
        else:
            self.chat_history.add_user_message(f"Viewed candidate: {event['candidate']}")
            self.chat_history.add_ai_message("Candidate interaction recorded")

    def _add(self, event: Dict[str, Any]) -> None:
        event["timestamp"] = datetime.now().isoformat(timespec="seconds")
        self.events.append(event)
        self._mirror(event)
        self._save()

    # -------------------------------------------------------------- recording
    def record_search(self, query: str, results_count: int) -> None:
        self._add({"type": "search", "query": query, "results_count": results_count})

    def record_candidate_view(self, candidate_name: str) -> None:
        self._add({"type": "view", "candidate": candidate_name})

    # -------------------------------------------------------------- retrieval
    def get_search_history(self, n: int = 5) -> List[str]:
        return [e["query"] for e in self.events if e["type"] == "search"][-n:]

    def get_recent_views(self, n: int = 5) -> List[str]:
        return [e["candidate"] for e in self.events if e["type"] == "view"][-n:]

    def get_memory_stats(self) -> Dict[str, int]:
        return {
            "total_messages": len(self.chat_history.messages),
            "search_count": sum(e["type"] == "search" for e in self.events),
            "candidate_views": sum(e["type"] == "view" for e in self.events),
        }

    def clear(self) -> None:
        self.events = []
        self.chat_history.clear()
        self._save()