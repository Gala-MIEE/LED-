"""Запуск оценки RAGAS из корня проекта.

Сам скрипт лежит в assistant_giga и читает данные относительно этой папки.
Оценка и ответы идут только через GigaChat.
"""

import os
import runpy
import sys
from pathlib import Path

api_dir = Path(__file__).resolve().parent / "assistant_giga"
os.chdir(api_dir)
sys.path.insert(0, str(api_dir))
runpy.run_path(str(api_dir / "evaluate_ragas.py"), run_name="__main__")
