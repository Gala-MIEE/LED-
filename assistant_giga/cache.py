"""
Модуль кеширования для RAG-ассистента.
Хранит пары вопрос-ответ в SQLite и умеет узнавать близкие формулировки.
"""

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from vector_store import cosine_similarity


DEFAULT_TTL_HOURS = 168
DEFAULT_MIN_SIMILARITY = 0.93


def _normalize_context(raw: Optional[str]) -> List[Dict[str, Any]]:
    """Приводит старый список строк и новый список фрагментов к одному виду."""
    if not raw:
        return []
    data = json.loads(raw)
    documents = []
    for item in data:
        if isinstance(item, str):
            documents.append({
                "id": "",
                "text": item,
                "distance": None,
                "source": "",
                "section": "",
            })
        elif isinstance(item, dict):
            distance = item.get("distance")
            documents.append({
                "id": item.get("id", ""),
                "text": item.get("text", ""),
                "distance": None if distance is None else float(distance),
                "source": item.get("source", ""),
                "section": item.get("section", ""),
            })
    return documents


def _context_payload(context: Optional[list]) -> Optional[str]:
    if not context:
        return None
    payload = []
    for item in context:
        if isinstance(item, str):
            payload.append({
                "id": "",
                "text": item,
                "distance": None,
                "source": "",
                "section": "",
            })
        elif isinstance(item, dict):
            distance = item.get("distance")
            payload.append({
                "id": item.get("id", ""),
                "text": item.get("text", ""),
                "distance": None if distance is None else float(distance),
                "source": item.get("source", "") or "",
                "section": item.get("section", "") or "",
            })
    return json.dumps(payload, ensure_ascii=False)


class RAGCache:
    """Кеш для хранения результатов RAG-запросов."""

    def __init__(self, db_path: str = "rag_cache.db", ttl_hours: float = None,
                 min_similarity: float = None):
        """
        Args:
            db_path: путь к файлу SQLite
            ttl_hours: срок жизни записи в часах. 0 отключает устаревание.
            min_similarity: порог косинусной близости для похожего вопроса
        """
        self.db_path = db_path
        if ttl_hours is None:
            ttl_hours = float(os.getenv("CACHE_TTL_HOURS", str(DEFAULT_TTL_HOURS)))
        if min_similarity is None:
            min_similarity = float(os.getenv("CACHE_MIN_SIMILARITY", str(DEFAULT_MIN_SIMILARITY)))
        self.ttl_hours = ttl_hours
        self.min_similarity = min_similarity
        self._init_db()

    @contextmanager
    def _connect(self):
        """Соединение закрывается сразу: на Windows иначе файл базы остаётся занят."""
        connection = sqlite3.connect(self.db_path)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _init_db(self):
        """Создаёт таблицу и добавляет колонку эмбеддинга в уже существующую базу."""
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    context TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding TEXT
                )
            """)
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(cache)")
            }
            if "embedding" not in columns:
                connection.execute("ALTER TABLE cache ADD COLUMN embedding TEXT")

    def _query_hash(self, query: str) -> str:
        normalized_query = " ".join(query.lower().strip().split())
        return hashlib.sha256(normalized_query.encode()).hexdigest()

    def _is_expired(self, created_at: Optional[str]) -> bool:
        if self.ttl_hours <= 0 or not created_at:
            return False
        try:
            created = datetime.fromisoformat(created_at)
        except ValueError:
            return False
        return datetime.now() - created > timedelta(hours=self.ttl_hours)

    def _row_to_result(self, row) -> Dict[str, Any]:
        return {
            "query": row[0],
            "answer": row[1],
            "context": _normalize_context(row[2]),
            "created_at": row[3],
            "from_cache": True,
        }

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Точное совпадение после нормализации регистра и пробелов."""
        query_hash = self._query_hash(query)
        with self._connect() as connection:
            row = connection.execute("""
                SELECT query, answer, context, created_at
                FROM cache
                WHERE query_hash = ?
            """, (query_hash,)).fetchone()
            if row is None:
                return None
            if self._is_expired(row[3]):
                connection.execute("DELETE FROM cache WHERE query_hash = ?", (query_hash,))
                return None
            return self._row_to_result(row)

    def get_similar(self, embedding: List[float], min_similarity: float = None) -> Optional[Dict[str, Any]]:
        """Ищет ближайшую формулировку того же вопроса по эмбеддингу."""
        if not embedding:
            return None
        threshold = self.min_similarity if min_similarity is None else min_similarity
        best_row = None
        best_score = threshold

        with self._connect() as connection:
            rows = connection.execute("""
                SELECT query_hash, query, answer, context, created_at, embedding
                FROM cache
                WHERE embedding IS NOT NULL
            """).fetchall()
            expired_hashes = []
            for query_hash, query, answer, context, created_at, raw_embedding in rows:
                if self._is_expired(created_at):
                    expired_hashes.append(query_hash)
                    continue
                try:
                    stored = json.loads(raw_embedding)
                except (TypeError, json.JSONDecodeError):
                    continue
                score = cosine_similarity(embedding, stored)
                if score >= best_score:
                    best_score = score
                    best_row = (query, answer, context, created_at)
            if expired_hashes:
                connection.executemany(
                    "DELETE FROM cache WHERE query_hash = ?",
                    [(query_hash,) for query_hash in expired_hashes],
                )

        if best_row is None:
            return None
        return self._row_to_result(best_row)

    def set(self, query: str, answer: str, context: list = None, embedding: List[float] = None):
        """Сохраняет ответ. Повтор того же вопроса заменяет запись."""
        query_hash = self._query_hash(query)
        embedding_json = json.dumps(embedding) if embedding else None
        with self._connect() as connection:
            connection.execute("""
                INSERT OR REPLACE INTO cache
                    (query_hash, query, answer, context, created_at, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                query_hash,
                query,
                answer,
                _context_payload(context),
                datetime.now().isoformat(),
                embedding_json,
            ))

    def clear(self):
        """Очистка всего кеша."""
        with self._connect() as connection:
            connection.execute("DELETE FROM cache")

    def get_stats(self) -> Dict[str, Any]:
        """Статистика кеша."""
        with self._connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            dates = connection.execute("SELECT MIN(created_at), MAX(created_at) FROM cache").fetchone()

        return {
            "total_entries": count,
            "oldest_entry": dates[0] if dates[0] else None,
            "newest_entry": dates[1] if dates[1] else None,
            "db_size_mb": os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0,
        }


if __name__ == "__main__":
    cache = RAGCache("test_cache.db", ttl_hours=0)
    cache.set(
        query="Что такое машинное обучение?",
        answer="Машинное обучение - это раздел искусственного интеллекта...",
        context=[{"text": "doc1", "section": "Определение"}],
        embedding=[1.0, 0.0],
    )
    print("Результат из кеша:", cache.get("Что такое машинное обучение?"))
    print("Похожий вопрос:", cache.get_similar([0.99, 0.01]))
    print("Статистика кеша:", cache.get_stats())
    if os.path.exists("test_cache.db"):
        os.remove("test_cache.db")
