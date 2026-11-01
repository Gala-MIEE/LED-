"""
Telegram-бот поверх того же RAG-ассистента.
В чат уходит только ответ. Фрагменты базы печатаются в консоли.
"""

import os
import sys
import time
from pathlib import Path

TELEGRAM_MESSAGE_LIMIT = 4000
_HISTORY_LIMIT = 6
_http = None


def answer_for_telegram(result: dict) -> str:
    """Текст для чата: ответ без фрагментов базы и без названия модели."""
    return (result.get("answer") or "").strip()


def split_telegram_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list:
    """Делит длинный ответ на части, которые принимает Telegram."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    parts = []
    rest = text
    while rest:
        if len(rest) <= limit:
            parts.append(rest)
            break
        cut = rest.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = rest.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        piece = rest[:cut].strip()
        if piece:
            parts.append(piece)
        rest = rest[cut:].strip()
    return parts


def remember_turn(histories: dict, chat_id: int, query: str, answer: str) -> list:
    """Хранит последние реплики отдельно для каждого чата."""
    history = histories.setdefault(chat_id, [])
    history.append({"query": query, "answer": answer})
    if len(history) > _HISTORY_LIMIT:
        del history[:-_HISTORY_LIMIT]
    return history


def advance_offset(offset, update_id: int, delivered: bool):
    """Сдвигает очередь Telegram только после того, как ответ удалось отправить."""
    if not delivered:
        return offset
    return update_id + 1


def _bind_session(session):
    global _http
    _http = session


def _connect_telegram(token: str):
    """Сначала локальный прокси Windows, затем прямое соединение.

    Прямой ответ api.telegram.org с этой сети приходит примерно за полминуты,
    поэтому короткое ожидание выглядит как полный отказ Telegram.
    """
    import requests
    from urllib.request import getproxies

    if getproxies():
        session = requests.Session()
        session.trust_env = True
        _bind_session(session)
        print("Соединение с Telegram через локальный прокси...")
        me = _telegram_call(token, "getMe", http_timeout=(20, 40), quiet=True)
        if me:
            return me
        print("Локальный прокси не ответил вовремя.")

    session = requests.Session()
    session.trust_env = False
    _bind_session(session)
    print("Прямое соединение с Telegram. Ответ может идти около минуты...")
    return _telegram_call(token, "getMe", http_timeout=(60, 60))


def _telegram_call(token: str, method: str, http_timeout, quiet: bool = False, **payload):
    """Вызов Bot API. В консоль не попадает адрес с ключом."""
    import requests

    session = _http if _http is not None else requests.Session()
    url = f"https://api.telegram.org/bot{token}/{method}"
    try:
        response = session.post(url, json=payload, timeout=http_timeout)
    except requests.Timeout:
        if method == "getUpdates":
            return []
        if not quiet:
            print(f"Telegram {method}: превышено время ожидания")
        return None
    except requests.RequestException:
        if not quiet:
            print(f"Нет связи с Telegram ({method}).")
        return None

    if response.status_code != 200:
        print(f"Telegram {method}: статус {response.status_code}")
        return None
    try:
        body = response.json()
    except ValueError:
        print(f"Telegram {method}: ответ не JSON")
        return None
    if not body.get("ok"):
        print(f"Telegram {method}: {body.get('description', 'неизвестная ошибка')}")
        return None
    return body.get("result")


def _send_payload(token: str, payload: dict) -> bool:
    """Повторяет отправку: до api.telegram.org отсюда ответ часто идёт дольше 20 секунд."""
    for attempt in range(3):
        sent = _telegram_call(
            token,
            "sendMessage",
            http_timeout=(60, 60),
            quiet=True,
            **payload,
        )
        if sent is not None:
            return True
        if attempt < 2:
            print("Telegram отвечает медленно, повторяю отправку...")
            time.sleep(2)
    print("Telegram sendMessage: не удалось отправить ответ")
    return False


def _send_answer(token: str, chat_id: int, message_id: int, text: str) -> bool:
    parts = split_telegram_message(text)
    if not parts:
        parts = ["Пустой ответ."]
    for index, part in enumerate(parts):
        payload = {"chat_id": chat_id, "text": part}
        if index == 0 and message_id:
            payload["reply_to_message_id"] = message_id
        if _send_payload(token, payload):
            continue
        if "reply_to_message_id" not in payload:
            return False
        payload.pop("reply_to_message_id")
        if not _send_payload(token, payload):
            return False
    return True


def _handle_message(pipeline, histories: dict, token: str, message: dict, print_response) -> bool:
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return True
    text = (message.get("text") or "").strip()
    message_id = message.get("message_id")
    command = text.split()[0].lower() if text.startswith("/") else ""
    command = command.split("@", 1)[0]

    if command in {"/start", "/help"}:
        print(f"Команда {command} из чата {chat_id}. Отправляю приветствие...")
        return _send_answer(
            token,
            chat_id,
            message_id,
            "Здравствуйте. Я ассистент службы поддержки «СветоТех».\n"
            "Напишите вопрос о светодиодной продукции.\n"
            "Команда /clear начинает диалог заново.",
        )

    if command == "/clear":
        histories[chat_id] = []
        print(f"\nДиалог Telegram-чата {chat_id} начат заново")
        return _send_answer(token, chat_id, message_id, "Диалог начат заново. Напишите новый вопрос.")

    if not text or text.startswith("/"):
        return _send_answer(token, chat_id, message_id, "Напишите вопрос текстом.")

    _telegram_call(
        token,
        "sendChatAction",
        http_timeout=(3, 5),
        quiet=True,
        chat_id=chat_id,
        action="typing",
    )
    history = histories.get(chat_id, [])
    try:
        result = pipeline.query(text, history=history)
    except Exception as error:
        print(f"\nОшибка ответа для чата {chat_id}: {error}")
        return _send_answer(
            token,
            chat_id,
            message_id,
            "Не получилось подготовить ответ. Попробуйте ещё раз через минуту.",
        )

    remember_turn(histories, chat_id, text, result["answer"])
    print(f"\nTelegram, чат {chat_id}")
    print_response(result)
    return _send_answer(token, chat_id, message_id, answer_for_telegram(result))


def _load_env():
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def main():
    """Запускает бота и печатает источники в этой консоли."""
    _load_env()
    from app import _configure_logging, print_response
    from gigachat_client import gigachat_credentials_set
    from rag_pipeline import RAGPipeline

    _configure_logging()
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        print("Ошибка: в .env нет TELEGRAM_BOT_TOKEN")
        sys.exit(1)
    if not gigachat_credentials_set():
        print("Ошибка: ключ GigaChat не установлен")
        sys.exit(1)

    me = _connect_telegram(token)
    if not me:
        sys.exit(1)
    username = me.get("username") or "бот"
    print(f"Бот @{username} запущен.")
    print("В Telegram уходят только ответы. Фрагменты базы печатаются здесь.")
    print("Остановка: Ctrl+C\n")

    print("Инициализация системы...\n")
    try:
        pipeline = RAGPipeline(
            collection_name="gigachat_rag_collection",
            cache_db_path="gigachat_rag_cache.db",
            data_file="data",
        )
    except Exception as error:
        print(f"Ошибка инициализации: {error}")
        sys.exit(1)
    print("\nБот готов принимать вопросы.\n")

    offset = None
    histories = {}
    while True:
        try:
            payload = {"timeout": 25}
            if offset is not None:
                payload["offset"] = offset
            updates = _telegram_call(token, "getUpdates", http_timeout=(60, 90), **payload)
            if not updates:
                continue
            for update in updates:
                message = update.get("message")
                if not message:
                    offset = advance_offset(offset, update["update_id"], True)
                    continue
                delivered = _handle_message(pipeline, histories, token, message, print_response)
                offset = advance_offset(offset, update["update_id"], delivered)
                if not delivered:
                    print("Ответ не доставлен. Тот же вопрос будет отправлен ещё раз.")
                    break
        except KeyboardInterrupt:
            print("\nБот остановлен.")
            break


if __name__ == "__main__":
    main()
