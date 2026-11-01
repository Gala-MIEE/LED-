"""
Консольное приложение для взаимодействия с RAG ассистентом.
"""

import logging
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from gigachat_client import gigachat_credentials_set
from rag_pipeline import RAGPipeline

# Загрузка переменных окружения из .env файла
# Ищем .env в корне проекта (на уровень выше)
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # Пытаемся загрузить из текущей директории
    load_dotenv()


def print_banner():
    """Вывод приветственного баннера."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║         RAG Ассистент (GigaChat Mode)                   ║
║  Retrieval-Augmented Generation с GigaChat от Сбера     ║
╚══════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("Введите 'exit' или 'quit' для выхода")
    print("Введите 'stats' для просмотра статистики")
    print("Введите 'clear' для очистки кеша\n")


def print_response(result: dict):
    """
    Форматированный вывод ответа.
    
    Args:
        result: словарь с результатом запроса
    """
    print(f"\n{'─'*60}")
    print(f"📝 Вопрос: {result['query']}")
    print(f"{'─'*60}")
    
    # Индикатор источника ответа
    if result['from_cache']:
        print("💾 Источник: КЕШ")
        if 'cached_at' in result:
            print(f"   Сохранено: {result['cached_at']}")
    else:
        print(f"🤖 Источник: {result.get('model', 'LLM')}")
        print(f"   Использовано документов: {len(result.get('context_docs', []))}")
    
    print(f"\n💬 Ответ:\n{result['answer']}")

    docs = result.get("context_docs") or []
    if docs:
        print("\n📚 Фрагменты базы знаний:")
        for i, doc in enumerate(docs[:3], 1):
            if isinstance(doc, str):
                preview = doc[:150] + "..." if len(doc) > 150 else doc
                print(f"   {i}. {preview}")
                continue
            preview = doc["text"][:150] + "..." if len(doc["text"]) > 150 else doc["text"]
            label_parts = [part for part in (doc.get("section"), doc.get("source")) if part]
            if label_parts:
                print(f"   {i}. {' — '.join(label_parts)}")
                print(f"      {preview}")
            else:
                print(f"   {i}. {preview}")
    elif not result["from_cache"]:
        print("\n📚 В базе нет фрагмента ближе порога похожести")
    
    print(f"{'─'*60}\n")


def print_stats(pipeline: RAGPipeline):
    """
    Вывод статистики системы.
    
    Args:
        pipeline: экземпляр RAG pipeline
    """
    stats = pipeline.get_stats()
    
    print(f"\n{'═'*60}")
    print("📊 СТАТИСТИКА СИСТЕМЫ")
    print(f"{'═'*60}")
    
    print("\n🗄️  Векторное хранилище:")
    print(f"   Коллекция: {stats['vector_store']['name']}")
    print(f"   Документов: {stats['vector_store']['count']}")
    print(f"   Директория: {stats['vector_store']['persist_directory']}")
    
    print("\n💾 Кеш:")
    print(f"   Записей: {stats['cache']['total_entries']}")
    print(f"   Размер БД: {stats['cache']['db_size_mb']:.2f} MB")
    if stats['cache']['oldest_entry']:
        print(f"   Первая запись: {stats['cache']['oldest_entry']}")
    if stats['cache']['newest_entry']:
        print(f"   Последняя запись: {stats['cache']['newest_entry']}")
    
    print(f"\n🤖 Модель: {stats['model']}")
    print(f"{'═'*60}\n")


def _configure_logging():
    """Служебные шаги запроса не смешиваются с ответом. Уровень: RAG_LOG_LEVEL."""
    level_name = os.getenv("RAG_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def main():
    """Главная функция приложения."""
    _configure_logging()
    print_banner()
    
    if not gigachat_credentials_set():
        print("❌ Ошибка: ключ GigaChat не установлен")
        print("\nДобавьте в файл .env в корне проекта:")
        print("  GIGACHAT_AUTHORIZATION_KEY=your-auth-key")
        print("  GIGACHAT_SCOPE=GIGACHAT_API_PERS")
        print("  GIGACHAT_MODEL=GigaChat")
        sys.exit(1)
    
    try:
        # Инициализация RAG pipeline
        print("🚀 Инициализация системы...\n")
        pipeline = RAGPipeline(
            collection_name="gigachat_rag_collection",
            cache_db_path="gigachat_rag_cache.db",
            data_file="data"
        )
        print("\n✅ Система готова к работе!\n")
        
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        sys.exit(1)
    
    history = []

    # Основной цикл взаимодействия
    while True:
        try:
            # Получение запроса от пользователя
            user_input = input("💭 Ваш вопрос: ").strip()
            
            # Обработка специальных команд
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 До свидания!")
                break
            
            if user_input.lower() == 'stats':
                print_stats(pipeline)
                continue
            
            if user_input.lower() == 'clear':
                confirm = input("⚠️  Вы уверены, что хотите очистить кеш? (yes/no): ")
                if confirm.lower() in ['yes', 'y', 'да']:
                    pipeline.cache.clear()
                    history.clear()
                    print("✅ Кеш очищен")
                continue
            
            if not user_input:
                print("⚠️  Пожалуйста, введите вопрос\n")
                continue
            
            result = pipeline.query(user_input, history=history)
            history.append({"query": user_input, "answer": result["answer"]})
            if len(history) > 6:
                history[:] = history[-6:]

            print_response(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}\n")


if __name__ == "__main__":
    main()

