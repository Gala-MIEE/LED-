"""
Основной RAG pipeline для GigaChat.
Поток: запрос → кеш → поиск с порогом → GigaChat → ответ → кеш.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from cache import RAGCache
from vector_store import VectorStore, project_path, resolve_data_file

logger = logging.getLogger("assistant_giga")

NO_INFO_ANSWER = "В базе знаний нет информации, чтобы ответить на этот вопрос."

_FOLLOWUP_PREFIXES = (
    "а ",
    "и ",
    "ну ",
    "это ",
    "этот ",
    "эта ",
    "там ",
    "тогда ",
    "ещё ",
    "еще ",
    "такой ",
    "такая ",
    "такое ",
    "тот ",
    "та ",
    "для него",
    "для неё",
    "для нее",
    "а для",
    "а если",
    "а можно",
    "а как",
    "а сколько",
    "а какой",
    "а какая",
    "а какие",
)


def is_followup(query: str) -> bool:
    """Короткое уточнение, которое само по себе не является новым вопросом."""
    text = " ".join(query.lower().split())
    return text.startswith(_FOLLOWUP_PREFIXES)


def retrieval_query(query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """Для уточнения добавляет предыдущий вопрос, чтобы поиск шёл по полной теме."""
    if not history or not is_followup(query):
        return query
    previous = (history[-1].get("query") or "").strip()
    if not previous:
        return query
    return f"{previous}. {query}"


def build_prompt(query: str, context_docs: List[Dict[str, Any]],
                 history: Optional[List[Dict[str, str]]] = None) -> str:
    """Промпт с фрагментами базы и несколькими предыдущими репликами."""
    context_parts = []
    for index, doc in enumerate(context_docs, 1):
        section = doc.get("section") or ""
        heading = f" ({section})" if section else ""
        context_parts.append(f"Документ {index}{heading}:\n{doc['text']}\n")
    context = "\n".join(context_parts)

    history_block = ""
    if history:
        turns = []
        for turn in history[-4:]:
            turns.append(f"Пользователь: {turn.get('query', '')}\nАссистент: {turn.get('answer', '')}")
        history_block = (
            "Предыдущий диалог (только чтобы понять уточнение):\n"
            + "\n\n".join(turns)
            + "\n\n"
        )

    return f"""Ты - полезный AI ассистент. Ответь на вопрос пользователя на основе предоставленного контекста.

Контекст:
{context}

{history_block}Вопрос: {query}

Инструкции:
- Отвечай только на основе предоставленного контекста
- Уточняющий вопрос понимай с учётом предыдущего диалога
- Если в контексте нет информации для ответа, скажи об этом
- Будь точным и кратким
- Отвечай на русском языке

Ответ:"""


class RAGPipeline:
    """Основной pipeline для RAG-системы с GigaChat."""

    def __init__(self,
                 collection_name: str = "rag_collection",
                 cache_db_path: str = "rag_cache.db",
                 data_file: str = "data",
                 model: str = None):
        """
        Инициализация RAG pipeline.

        Args:
            collection_name: имя коллекции в ChromaDB
            cache_db_path: путь к базе данных кеша
            data_file: путь к документу Word или к папке data
            model: модель GigaChat для генерации ответов
        """
        if not os.getenv("GIGACHAT_AUTHORIZATION_KEY") and not os.getenv("GIGACHAT_AUTH_KEY"):
            raise ValueError(
                "Ключ GigaChat не установлен. Добавьте GIGACHAT_AUTHORIZATION_KEY в .env"
            )

        self.model = model or os.getenv("GIGACHAT_MODEL") or "GigaChat"
        from gigachat_client import GigaChatClient
        self.gigachat_client = GigaChatClient()

        print("Инициализация векторного хранилища...")
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=project_path("chroma_db"),
        )

        data_path = resolve_data_file(project_path(data_file))
        print(f"Загрузка базы знаний из {data_path}...")
        reloaded = self.vector_store.load_documents(data_path)

        print("Инициализация кеша...")
        self.cache = RAGCache(db_path=project_path(cache_db_path))
        if reloaded:
            self.cache.clear()
            print("Индекс базы знаний собран заново, старые ответы в кеше удалены")

        print("RAG Pipeline инициализирован (GigaChat)")

    def _generate_answer(self, prompt: str) -> str:
        """Генерация ответа через GigaChat."""
        messages = [
            {
                "role": "system",
                "content": "Ты - полезный AI ассистент, который отвечает на вопросы на основе предоставленного контекста.",
            },
            {"role": "user", "content": prompt},
        ]
        answer = self.gigachat_client.chat_completion(
            messages=messages,
            model=self.model,
            temperature=0.3,
            max_tokens=500,
        )
        return answer.strip()

    def _cached_response(self, user_query: str, cached_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "query": user_query,
            "answer": cached_result["answer"],
            "from_cache": True,
            "context_docs": cached_result.get("context") or [],
            "cached_at": cached_result.get("created_at"),
            "model": self.model,
        }

    def query(self, user_query: str, use_cache: bool = True,
              history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Обрабатывает вопрос пользователя.

        Сначала точный кеш, затем похожая формулировка, затем поиск.
        Если близких фрагментов нет, модель не вызывается.
        Уточнения ищутся вместе с предыдущим вопросом.
        """
        history = history or []
        lookup = retrieval_query(user_query, history)
        logger.debug("Запрос: %s", lookup)

        query_embedding = None
        if use_cache:
            cached_result = self.cache.get(lookup)
            if cached_result is None:
                query_embedding = self.vector_store.embed_query(lookup)
                cached_result = self.cache.get_similar(query_embedding)
            if cached_result:
                logger.debug("Ответ взят из кеша")
                return self._cached_response(user_query, cached_result)

        if query_embedding is None:
            query_embedding = self.vector_store.embed_query(lookup)

        context_docs = self.vector_store.search(
            lookup,
            top_k=3,
            query_embedding=query_embedding,
        )
        logger.debug("Фрагментов после порога: %s", len(context_docs))

        if not context_docs:
            return {
                "query": user_query,
                "answer": NO_INFO_ANSWER,
                "from_cache": False,
                "context_docs": [],
                "model": self.model,
            }

        prompt = build_prompt(user_query, context_docs, history)
        logger.debug("Генерация ответа через %s", self.model)
        answer = self._generate_answer(prompt)

        if use_cache:
            self.cache.set(lookup, answer, context_docs, query_embedding)

        return {
            "query": user_query,
            "answer": answer,
            "from_cache": False,
            "context_docs": context_docs,
            "model": self.model,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Статистика системы."""
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self.cache.get_stats(),
            "model": self.model,
        }


if __name__ == "__main__":
    import sys

    try:
        pipeline = RAGPipeline()
        test_queries = [
            "Можно ли подключать светодиодную ленту 24 В к блоку питания 12 В?",
            "Какая степень защиты IP нужна для ленты в ванной?",
            "Как оформить возврат светодиодной продукции?",
        ]
        for query in test_queries:
            result = pipeline.query(query)
            print(f"\n{'=' * 60}")
            print(f"Вопрос: {result['query']}")
            print(f"Из кеша: {result['from_cache']}")
            print(f"Ответ: {result['answer']}")
            print(f"{'=' * 60}\n")

        print("\n--- Повторный запрос ---")
        result = pipeline.query(test_queries[0])
        print(f"Из кеша: {result['from_cache']}")

        stats = pipeline.get_stats()
        print("\nСтатистика системы:")
        print(f"Векторное хранилище: {stats['vector_store']}")
        print(f"Кеш: {stats['cache']}")
    except Exception as error:
        print(f"Ошибка: {error}")
        sys.exit(1)
