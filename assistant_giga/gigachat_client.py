"""
Клиент для работы с GigaChat API от Сбера.
Управляет авторизацией и запросами к API.
"""

import requests
import os
import sys
import uuid
import warnings

for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json

try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
    else:
        load_dotenv()
except ImportError:
    pass

# В запросах к Сберу отключена проверка сертификата.
warnings.filterwarnings("ignore", message="Unverified HTTPS request")


def _env_value(*names: str) -> Optional[str]:
    """Первое непустое значение среди переменных окружения."""
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def gigachat_credentials_set() -> bool:
    """Есть ли ключ авторизации GigaChat в окружении."""
    return _env_value("GIGACHAT_AUTHORIZATION_KEY", "GIGACHAT_AUTH_KEY") is not None


class GigaChatClient:
    """Клиент для работы с GigaChat API."""
    
    def __init__(self, auth_key: str = None, rq_uid: str = None, scope: str = None):
        """
        Инициализация GigaChat клиента.
        
        Args:
            auth_key: ключ авторизации (Basic token)
            rq_uid: уникальный идентификатор запроса. Если не задан, создаётся UUID.
            scope: область доступа, по умолчанию из GIGACHAT_SCOPE
        """
        self.auth_key = auth_key or _env_value(
            "GIGACHAT_AUTHORIZATION_KEY", "GIGACHAT_AUTH_KEY"
        )
        self.rq_uid = rq_uid or _env_value("GIGACHAT_RQUID") or str(uuid.uuid4())
        self.scope = scope or _env_value("GIGACHAT_SCOPE") or "GIGACHAT_API_PERS"
        self.model = _env_value("GIGACHAT_MODEL") or "GigaChat"
        
        if self.auth_key and self.auth_key.lower().startswith("basic "):
            self.auth_key = self.auth_key.split(" ", 1)[1].strip()
        
        if not self.auth_key:
            raise ValueError(
                "Ключ GigaChat не установлен. Добавьте GIGACHAT_AUTHORIZATION_KEY в .env"
            )
        
        self.oauth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        self.api_url = "https://gigachat.devices.sberbank.ru/api/v1"
        
        self.access_token = None
        self.token_expires_at = None
        
        # Получаем токен при инициализации
        self._refresh_token()
    
    def _refresh_token(self):
        """Получение нового access token."""
        payload = {'scope': self.scope}
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': self.rq_uid,
            'Authorization': f'Basic {self.auth_key}'
        }
        
        try:
            response = requests.post(
                self.oauth_url,
                headers=headers,
                data=payload,
                verify=False  # Для работы с сертификатом Сбера
            )
            response.raise_for_status()
            
            data = response.json()
            self.access_token = data['access_token']
            # Токен действует 30 минут, обновляем за 1 минуту до истечения
            self.token_expires_at = datetime.now() + timedelta(minutes=29)
            
            print("✓ GigaChat access token получен")
            
        except Exception as e:
            raise Exception(f"Ошибка получения access token: {e}")
    
    def _ensure_token_valid(self):
        """Проверка валидности токена и обновление при необходимости."""
        if not self.access_token or datetime.now() >= self.token_expires_at:
            print("⟳ Обновление access token...")
            self._refresh_token()
    
    def _get_headers(self) -> Dict[str, str]:
        """Получение заголовков для запросов."""
        self._ensure_token_valid()
        return {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       model: str = None,
                       temperature: float = 0.3,
                       max_tokens: int = 500) -> str:
        """
        Отправка запроса к чат-модели GigaChat.
        
        Args:
            messages: список сообщений в формате [{"role": "user", "content": "..."}]
            model: название модели
            temperature: температура генерации
            max_tokens: максимальное количество токенов в ответе
            
        Returns:
            текст ответа от модели
        """
        url = f"{self.api_url}/chat/completions"
        
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                verify=False
            )
            response.raise_for_status()
            
            data = response.json()
            return data['choices'][0]['message']['content']
            
        except Exception as e:
            raise Exception(f"Ошибка запроса к GigaChat: {e}")
    
    def get_models(self) -> List[Dict[str, Any]]:
        """
        Получение списка доступных моделей.
        
        Returns:
            список моделей
        """
        url = f"{self.api_url}/models"
        
        try:
            response = requests.get(
                url,
                headers=self._get_headers(),
                verify=False
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get('data', [])
            
        except Exception as e:
            print(f"Ошибка получения списка моделей: {e}")
            return []


if __name__ == "__main__":
    # Тестирование клиента
    import sys
    from pathlib import Path
    from dotenv import load_dotenv
    
    # Загружаем .env
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    
    try:
        client = GigaChatClient()
        
        # Тест получения моделей
        print("\n=== Доступные модели ===")
        models = client.get_models()
        for model in models:
            print(f"- {model.get('id', 'unknown')}")
        
        # Тест чата
        print("\n=== Тест чата ===")
        response = client.chat_completion(
            messages=[
                {"role": "user", "content": "Что такое машинное обучение? Ответь кратко."}
            ]
        )
        print(f"Ответ: {response}")
        
        print("\n✓ Все тесты пройдены")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)

