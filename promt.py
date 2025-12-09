# promt.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import os

import requests
import random
import time


# ==========================
# 1. МОДЕЛИ ДАННЫХ
# ==========================

@dataclass
class AdVariant:
    channel: str
    headline: str
    text: str
    cta: str
    notes: Optional[str] = None


# ==========================
# 2. SYSTEM PROMPT (Общий)
# ==========================

SYSTEM_PROMPT = """
Ты — модуль генерации рекламных креативов для ИИ-платформы GENAI-4.
Твоя задача — создавать эффективные рекламные тексты для интернет-магазина электроники, адаптированные под разные каналы (Telegram, VK, Yandex Ads).
Сосредоточься на конверсии (кликах и покупках). Не используй лишнего текста, только то, что помогает продавать.

=====================
ОБЩИЕ ПРАВИЛА
=====================
1. Пиши только на русском языке.
2. Формируй тексты современно, понятно, без канцелярита.
3. Не придумывай технических характеристик, которых нет во входных данных.
4. Подчеркивай выгоды товара, а не только параметры.
5. Учитывай тренды маркетинга:
   - "минимализм" → краткость, сухая подача выгоды
   - "FOMO" → ограниченность, «успей», «мало осталось»
   - "честность" → без преувеличений
   - "социальное доказательство" → популярность, отзывы
   - "юмор" → легкий, не кринж
6. Строго соблюдай требования канала (см. ниже).
7. Отвечай ТОЛЬКО JSON-структурой, без текста вне JSON.

=====================
ВХОДНЫЕ ДАННЫЕ
=====================
Ты получаешь JSON следующего вида:
{
  "product": {
    "name": "Название товара",
    "category": "Категория",
    "price": 12345,
    "tags": ["новинка", "яркий"],
    "features": ["Характеристика 1", "Характеристика 2"],
    "recommendation": "Краткое summary от анализатора"
  },
  "audience_profile": {
    "age_range": "18-30",
    "interests": ["Технологии"],
    "behavior": ["Реагирует на скидки"]
  },
  "channel": "telegram" | "vk" | "yandex_ads",
  "n_variants": 3
}

=====================
ТРЕБОВАНИЯ КАНАЛОВ
=====================
- **telegram**: Максимальная краткость (до 150 символов), активное использование эмодзи, создание ощущения срочности (FOMO). Фокус на одном ключевом преимуществе.
- **vk**: Длинное, подробное описание (до 300-500 символов), социальное доказательство (популярность, отзывы), создание доверия и полноты информации.
- **yandex_ads**: Заголовок (до 56 символов) и текст (до 81 символа). Сухо, по делу, только ключевые выгоды и характеристики, под поисковый запрос.

=====================
ФОРМАТ ВЫВОДА (ТОЛЬКО JSON)
=====================
Тебе необходимо сгенерировать n_variants (3) вариантов креатива.

{
  "variants": [
    {
      "channel": "telegram",
      "headline": "Твой новый заголовок",
      "text": "Текст с эмодзи",
      "cta": "Призыв к действию",
      "notes": "Краткий комментарий, почему этот креатив сработает."
    },
    // ... еще два таких объекта
  ]
}
"""


# ==========================
# 3. MISTRAL API CLIENT
# ==========================

class MistralClient:
    """Клиент реального Mistral API (chat.completions)."""

    def __init__(self, model: str = "mistral-large"):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            # Ошибка, если ключа нет, чтобы main.py мог переключиться на Mock
            raise ValueError("MISTRAL_API_KEY не задан в переменных окружения.")

        self.api_key = api_key
        self.model = model
        # Используем официальный эндпоинт Mistral
        self.base_url = "https://api.mistral.ai/v1/chat/completions"

    def _call_api(self, messages: List[Dict[str, str]], response_format: str = "json") -> Dict[str, Any]:
        """Универсальный метод для вызова API Mistral."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.9,
            "response_format": {"type": "json_object"} if response_format == "json" else None,
        }

        if data['response_format'] is None:
            del data['response_format']

        try:
            # Используем requests для синхронного вызова (стандартно для LLM-клиентов)
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()  # Вызывает HTTPError для ошибок 4xx/5xx
            return response.json()
        except requests.exceptions.RequestException as e:
            # Обработка ошибок API и сети
            print(f"Mistral API Error: {e}")
            raise ConnectionError(f"Ошибка при вызове Mistral API. Проверьте ключ и баланс. Детали: {e}")

    def generate_variants(self, payload: Dict[str, Any]) -> List[AdVariant]:
        """Генерирует N вариантов креатива, вызывая Mistral API."""
        product = payload["product"]
        channel = payload["channel"]
        n_variants = payload.get("n_variants", 1)

        input_data = {
            "product": product,
            "audience_profile": payload["audience_profile"],
            "channel": channel,
            "n_variants": n_variants,
        }

        # Собираем промпт, включая все входные данные
        user_prompt = f"Сгенерируй {n_variants} вариантов рекламного креатива. Строго соблюдай SYSTEM_PROMPT. Входной JSON: {json.dumps(input_data, ensure_ascii=False, indent=2)}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw_response = self._call_api(messages, response_format="json")
        except (ConnectionError, ValueError) as e:
            # На ошибке клиента или соединения — используем заглушку
            print(f"Mistral API недоступен или ошибка: {e}. Переключение на Mock.")
            return MockLLMClient().generate_variants(payload)

        try:
            # Парсинг ответа
            raw_content = raw_response["choices"][0]["message"]["content"]
            parsed_json = json.loads(raw_content)

            variants_data = parsed_json.get("variants", [])

            variants: List[AdVariant] = []
            for v_data in variants_data:
                # Создаем объекты AdVariant
                variants.append(
                    AdVariant(
                        channel=v_data.get("channel", channel),
                        headline=v_data.get("headline", "Заголовок от Mistral"),
                        text=v_data.get("text", "Текст от Mistral"),
                        cta=v_data.get("cta", "CTA"),
                        notes=v_data.get("notes", "Сгенерировано Mistral"),
                    )
                )

            # Если получили меньше, чем просили, заполняем заглушкой
            if len(variants) < n_variants:
                print(f"Mistral вернул {len(variants)} вариантов, но просили {n_variants}. Добавляем Mock.")
                mock_variants = MockLLMClient().generate_variants(payload)
                variants.extend(mock_variants)

            return variants[:n_variants]

        except Exception as e:
            print(f"Критическая ошибка парсинга или ответа Mistral: {e}. Ответ: {raw_content[:200]}")
            # На ошибке парсинга — возвращаем заглушку
            return MockLLMClient().generate_variants(payload)


# ==========================
# 4. MOCK (ЗАГЛУШКА) CLIENT
# ==========================

class MockLLMClient:
    """Заглушка для LLM-клиента (используется, если API недоступен)."""

    def __init__(self, model: str = "mock-model"):
        self.model = model

    def generate_variants(self, payload: Dict[str, Any]) -> List[AdVariant]:
        """Генерирует N вариантов, используя простые шаблоны."""
        product = payload["product"]
        channel = payload["channel"]
        name = product.get("name", "Товар")
        features = product.get("features") or []
        features_text = ", ".join([f for f in features if f]) or "отличные характеристики"

        if channel == "telegram":
            base = AdVariant(
                channel="telegram",
                headline=f"🔥 {name} — забери, пока есть!",
                text=f"Наш {name} с {features_text}. Это новинка, которую все ждут. Успей, пока цена ещё держится! 🚀",
                cta="Успеть взять сейчас →",
                notes="Mock: кратко, эмоции, FOMO.",
            )
        elif channel == "vk":
            base = AdVariant(
                channel="vk",
                headline=f"{name}: Техника, которая радует каждый день | Отзывы 4.9/5",
                text=(f"{name} — выбор тех, кто ценит комфорт и качество. "
                      f"Особенности: {features_text}. Посмотрите, что говорят другие покупатели! Многие уже оценили этот вариант."),
                cta="Заказать онлайн и получить скидку",
                notes="Mock: длиннее текст, соцдоказательство.",
            )
        else:
            base = AdVariant(
                channel="yandex_ads",
                headline=f"Выгодная Цена на {name} — Спешите!",
                text=f"{name} с {features_text}. Быстрая доставка по РФ. Гарантия 1 год. Заказать онлайн.",
                cta="Купить онлайн",
                notes="Mock: сухо, по делу, под поиск.",
            )

        n = payload.get("n_variants", 1)
        # Возвращаем N копий базового варианта
        return [AdVariant(
            channel=base.channel,
            headline=f"{base.headline} (Вариант {i + 1})",
            text=base.text,
            cta=base.cta,
            notes=base.notes,
        ) for i in range(n)]


# ==========================
# 5. ФОРМАТИРОВАНИЕ ДЛЯ ВЫВОДА
# ==========================

def format_variant_for_channel(variant: AdVariant) -> str:
    """Форматирует креатив для удобного чтения."""
    ch = variant.channel.lower()
    if ch == "telegram":
        return (
            f"Telegram\n\n"
            f"{variant.headline}\n"
            f"{variant.text}\n"
            f"⬇️ {variant.cta}\n"
        )
    elif ch == "vk":
        return (
            f"VK\n\n"
            f"Заголовок: {variant.headline}\n"
            f"Текст:\n{variant.text}\n"
            f"[Кнопка: {variant.cta}]\n"
        )
    elif ch == "yandex_ads":
        return (
            f"Yandex Ads\n\n"
            f"Заголовок: {variant.headline}\n"
            f"Текст: {variant.text}\n"
            f"[CTA: {variant.cta}]\n"
        )
    else:
        return (
            f"{variant.channel}\n\n"
            f"{variant.headline}\n"
            f"{variant.text}\n"
            f"{variant.cta}\n"
        )


def format_all_variants_human_readable(variants: List[AdVariant]) -> List[str]:
    return [format_variant_for_channel(v) for v in variants]
