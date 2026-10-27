# RAG-ассистент службы поддержки

Ассистент отвечает на вопросы по базе знаний в Word: в консоли или в Telegram. Поиск идёт по локальным эмбеддингам, ответ формирует GigaChat. Повтор того же или очень близкого вопроса берётся из кеша.

## Установка

Нужен Python 3.11.

```powershell
winget install Python.Python.3.11
py -3.11 -m venv venv_py311
.\venv_py311\Scripts\activate
pip install -r .\requirements.txt
```

Ключи лежат в `.env` в корне проекта:

- `GIGACHAT_AUTHORIZATION_KEY`
- `GIGACHAT_SCOPE`
- `GIGACHAT_MODEL`
- `TELEGRAM_BOT_TOKEN`

## Запуск

Из папки `assistant_giga`:

```powershell
python app.py
python telegram_bot.py
```

Из корня проекта те же команды длиннее: `python assistant_giga\app.py` и `python assistant_giga\telegram_bot.py`.

Консоль принимает `stats`, `clear`, `exit`. В Telegram бот отвечает только текстом ответа, фрагменты базы печатаются в консоли, где он запущен. `/start` — приветствие, `/clear` — новый диалог в этом чате.

Уточнение вроде «а для улицы?» связывается с предыдущим вопросом. Если в базе нет близкого фрагмента, ассистент так и пишет и не вызывает модель.

Оценка качества:

```powershell
python evaluate_ragas.py
```

Проверки без сети и без ключа:

```powershell
python assistant_giga\tests\test_core.py
```

## Настройки

| Переменная | По умолчанию | Зачем |
| --- | --- | --- |
| `RAG_MAX_DISTANCE` | `0.65` | Максимальная косинусная дистанция фрагмента. Меньше — строже отбор. |
| `CACHE_TTL_HOURS` | `168` | Срок жизни ответа в кеше, в часах. `0` отключает устаревание. |
| `CACHE_MIN_SIMILARITY` | `0.93` | Насколько вопрос должен совпасть с уже заданным, чтобы взять кеш. |
| `RAG_LOG_LEVEL` | `WARNING` | `DEBUG` показывает шаги обработки запроса. |

База знаний — файл `.docx` в `assistant_giga\data`. При его изменении индекс собирается заново, а старые ответы в кеше удаляются.
