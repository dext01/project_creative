# promt.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # на всякий случай, чтобы не падать, если lib не установлена


SYSTEM_PROMPT = """
Ты — модуль генерации рекламных креативов для ИИ-платформы GENAI-4.
Твоя задача — создавать эффективные рекламные тексты для интернет-магазина электроники,
адаптированные под разные каналы (Telegram, VK, Yandex Ads).

Фокус: максимальная конверсия (клик / покупка).

ОБЩИЕ ПРАВИЛА:
- Пиши только на русском языке.
- Не придумывай характеристик, которых нет в описании товара.
- Подчеркивай выгоды и понятные пользователю результаты.
- Учитывай тренды: минимализм, честность, FOMO, социальное доказательство, юмор (легкий).

ФОРМАТ ВХОДА (один объект в JSON):
{
  "product": {
    "name": "...",
    "category": "...",
    "price": 12345,
    "margin": "высокая" или число или null,
    "tags": ["новинка", "яркий", "bestseller"],
    "features": ["описание", "характеристики"]
  },
  "audience_profile": {
    "age_range": "18-30",
    "interests": [...],
    "behavior": [...]
  },
  "channel": "telegram" | "vk" | "yandex_ads",
  "trends": ["минимализм", "FOMO", ...],
  "n_variants": 3
}

ТВОЯ ЗАДАЧА:
- сгенерировать n_variants объявлений для одного канала и одного товара.

ТРЕБОВАНИЯ К КАНАЛАМ:

[TELEGRAM]
- Короткий, эмоциональный текст.
- Заголовок до ~50 символов.
- 1–3 предложения, можно эмодзи (до 5 шт).
- FOMO приветствуется.
- CTA: "Успеть взять сейчас", "Смотреть в каталоге", "Перейти к покупке".

[VK]
- 2–5 предложений, можно 1–2 абзаца.
- Легкий сторителлинг, социальное доказательство ("покупатели выбирают", "отзывы").
- CTA: "Заказать онлайн", "Узнать цену", "Смотреть характеристики".

[Yandex Ads]
- Сухо, конкретно, без эмодзи.
- Короткий заголовок с выгодой.
- 1–2 предложения, ключевые слова (доставка, скидка, купить онлайн).
- CTA: "Купить онлайн", "Заказать с доставкой", "Смотреть в магазине".

ФОРМАТ ВЫХОДА:
Верни строго JSON:

{
  "variants": [
    {
      "channel": "<канал>",
      "headline": "<заголовок>",
      "text": "<основной текст>",
      "cta": "<призыв>",
      "notes": "<почему это должно конвертировать>"
    },
    ...
  ]
}

Никакого текста вне JSON.
"""


@dataclass
class AdVariant:
    channel: str
    headline: str
    text: str
    cta: str
    notes: str


class LLMClient:
    """Клиент реального OpenAI API (chat.completions)."""

    def __init__(self, model: str = "gpt-4.1-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY не задан в переменных окружения.")
        if OpenAI is None:
            raise ImportError("Библиотека openai не установлена. pip install openai")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_variants(self, payload: Dict[str, Any]) -> List[AdVariant]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ],
            temperature=0.9,
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        variants: List[AdVariant] = []
        for v in data.get("variants", []):
            variants.append(
                AdVariant(
                    channel=v.get("channel", "").strip(),
                    headline=v.get("headline", "").strip(),
                    text=v.get("text", "").strip(),
                    cta=v.get("cta", "").strip(),
                    notes=v.get("notes", "").strip(),
                )
            )
        return variants


class MockLLMClient:
    """Заглушка, если нет ключа OpenAI: простые шаблонные тексты."""

    def __init__(self):
        pass

    def generate_variants(self, payload: Dict[str, Any]) -> List[AdVariant]:
        product = payload["product"]
        channel = payload["channel"]
        name = product.get("name", "Товар")
        features = product.get("features") or []
        features_text = ", ".join([f for f in features if f]) or "отличные характеристики"

        if channel == "telegram":
            base = AdVariant(
                channel="telegram",
                headline=f"{name} — забери, пока есть",
                text=f"{name} с {features_text}. Успей, пока цена ещё держится 🔥",
                cta="Успеть взять сейчас",
                notes="Mock: кратко, эмоции, FOMO.",
            )
        elif channel == "vk":
            base = AdVariant(
                channel="vk",
                headline=f"{name}: техника, которая радует каждый день",
                text=(f"{name} — выбор тех, кто ценит комфорт и качество. "
                      f"Особенности: {features_text}. Многие покупатели уже оценили этот вариант."),
                cta="Заказать онлайн",
                notes="Mock: длиннее текст, соцдоказательство.",
            )
        else:
            base = AdVariant(
                channel="yandex_ads",
                headline=f"{name} — выгодная цена",
                text=f"{name} с {features_text}. Быстрая доставка, заказать онлайн.",
                cta="Купить онлайн",
                notes="Mock: сухо, по делу, под поиск.",
            )

        n = payload.get("n_variants", 1)
        return [base for _ in range(n)]
