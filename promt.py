from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import os

from openai import OpenAI
from main import evaluate_ad  # оценщик (клики / покупки)


# ==========================
# 1. SYSTEM PROMPT
# ==========================

SYSTEM_PROMPT = """
Ты — модуль генерации рекламных креативов для ИИ-платформы GENAI-4.
Твоя задача — создавать эффективные рекламные тексты для интернет-магазина электроники,
адаптированные под разные каналы (Telegram, VK, Yandex Ads).
Отвечай строго JSON.
"""


# ==========================
# 2. DATA-MODEL
# ==========================

@dataclass
class Product:
    name: str
    category: str
    price: Optional[float] = None
    margin: Optional[str] = None
    tags: Optional[List[str]] = None
    features: Optional[List[str]] = None


@dataclass
class AudienceProfile:
    age_range: str
    interests: List[str]
    behavior: List[str]


@dataclass
class GenerationRequest:
    product: Product
    audience_profile: AudienceProfile
    channel: str
    trends: List[str]
    n_variants: int = 1


@dataclass
class AdVariant:
    channel: str
    headline: str
    text: str
    cta: str
    notes: str


# ==========================
# 3. LLM CLIENT (SAFE)
# ==========================

class LLMClient:
    def __init__(self, model: str = "gpt-4.1-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY не задан в переменных окружения!")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_variants(self, payload: Dict[str, Any]) -> List[AdVariant]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ]
        )

        data = json.loads(response.choices[0].message.content)

        variants = []
        for v in data.get("variants", []):
            variants.append(AdVariant(**v))
        return variants


class MockLLMClient:
    def generate_variants(self, payload: Dict[str, Any]) -> List[AdVariant]:
        p = payload["product"]
        ch = payload["channel"]

        if ch == "telegram":
            return [AdVariant(
                channel="telegram",
                headline=f"{p['name']} — забери сейчас",
                text=f"{p['name']} с выгодной ценой. Успей!",
                cta="Успеть взять",
                notes="Mock TG"
            )]

        if ch == "vk":
            return [AdVariant(
                channel="vk",
                headline=f"{p['name']} для повседневного комфорта",
                text=f"{p['name']} с отличным балансом цены и качества.",
                cta="Заказать онлайн",
                notes="Mock VK"
            )]

        return [AdVariant(
            channel="yandex_ads",
            headline=f"{p['name']} со скидкой",
            text=f"{p['name']} в наличии. Быстрая доставка.",
            cta="Купить онлайн",
            notes="Mock Yandex"
        )]


# ==========================
# 4. GENERATOR
# ==========================

class AdGenerator:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate(self, input_json: Dict[str, Any]) -> List[AdVariant]:
        return self.llm_client.generate_variants(input_json)


# ==========================
# 5. OPTIMIZATION
# ==========================

def optimize_ad(generator, input_json, audience_segment):
    best = None
    best_score = -1

    for _ in range(5):
        variants = generator.generate(input_json)
        for v in variants:
            text = f"{v.headline}\n{v.text}\n{v.cta}"
            score = evaluate_ad(text, audience_segment)["click_probability"]
            if score > best_score:
                best_score = score
                best = (v, score)

    return best


# ==========================
# 6. BUILD FINAL CAMPAIGN
# ==========================

def build_campaign(best_products_file="best_products.json",
                   output="final_campaign.json",
                   audience_segment="Low_income_pragmatic_youth"):

    with open(best_products_file, "r", encoding="utf-8") as f:
        best_products = json.load(f)

    generator = AdGenerator(MockLLMClient())

    audience_profile = {
        "age_range": "20-35",
        "interests": ["гаджеты", "онлайн-покупки", "скидки"],
        "behavior": ["реагирует на скидки"]
    }

    campaigns = []

    for product in best_products:
        for channel in ["telegram", "vk", "yandex_ads"]:

            input_json = {
                "product": product,
                "audience_profile": audience_profile,
                "channel": channel,
                "trends": ["минимализм", "FOMO"],
                "n_variants": 3
            }

            best_variant, score = optimize_ad(generator, input_json, audience_segment)

            campaigns.append({
                "product": {
                    "name": product["name"],
                    "category": product["category"],
                    "price": product.get("price")
                },
                "channel": channel,
                "ad": {
                    "headline": best_variant.headline,
                    "text": best_variant.text,
                    "cta": best_variant.cta,
                    "notes": best_variant.notes
                },
                "evaluation": {
                    "click_probability": score
                }
            })

    final = {
        "platform": "GENAI-4",
        "description": "Финальная рекламная кампания",
        "campaigns": campaigns
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=4)

    print("✅ Кампания создана:", output)


# ==========================
# 7. MAIN
# ==========================

if __name__ == "__main__":
    build_campaign()