import json
import random
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

from promt import MistralClient, MockLLMClient, AdVariant


# ==========================
# 0. ЗАГРУЗКА КАТАЛОГА + СКОРИНГ ТОВАРОВ
# ==========================


def load_catalog_from_filelike(file_like) -> List[Dict[str, Any]]:
    """
    Загрузка каталога из JSON/CSV в список словарей.
    Поддерживает формат:
    - [ {product...}, ... ]
    - { "products": [ ... ] }
    """
    name = getattr(file_like, "name", "").lower()
    if name.endswith(".json"):
        data = json.load(file_like)
        if isinstance(data, dict) and "products" in data:
            data = data["products"]
        return data
    else:
        df = pd.read_csv(file_like)
        return df.to_dict(orient="records")


def _compute_margin_score(product: Dict[str, Any]) -> float:
    price = float(product.get("price", 0) or 0)
    market_cost = product.get("market_cost")
    margin_field = product.get("margin")

    if isinstance(margin_field, (int, float)):
        margin_percent = float(margin_field)
    elif price > 0 and market_cost is not None:
        margin_percent = (price - float(market_cost)) / price * 100
    else:
        margin_percent = 30.0

    return max(0.0, min(1.0, margin_percent / 80.0))


def _compute_tag_score(product: Dict[str, Any]) -> float:
    tags = product.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]

    text = " ".join(tags).lower()
    score = 0.0
    if any(k in text for k in ["новинка", "new", "2024"]):
        score += 0.3
    if any(k in text for k in ["яркий", "bright", "цветной", "дизайн"]):
        score += 0.2
    if any(k in text for k in ["bestseller", "хит", "hit", "топ"]):
        score += 0.3
    return max(0.0, min(1.0, score))


def _compute_visual_score(product: Dict[str, Any]) -> float:
    desc = str(product.get("description") or "") + " " + str(product.get("category") or "")
    text = desc.lower()
    score = 0.0
    if any(k in text for k in ["rgb", "подсветка", "amoled", "красив", "дизайн"]):
        score += 0.4
    if any(k in text for k in ["компакт", "минимализм", "тонкий"]):
        score += 0.2
    return max(0.0, min(1.0, score))


def compute_product_ad_score(product: Dict[str, Any]) -> float:
    """
    Хитрая, но быстрая эвристика "насколько товар хорош для рекламы":
    комбинация маржи, тегов и визуальной привлекательности.
    """
    m = _compute_margin_score(product)
    t = _compute_tag_score(product)
    v = _compute_visual_score(product)
    return round((m * 0.5 + t * 0.3 + v * 0.2), 3)


def select_top_products(catalog: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    """
    Выбирает k лучших товаров по внутреннему рекламному скору.
    """
    scored = []
    for p in catalog:
        score = compute_product_ad_score(p)
        scored.append({**p, "_ad_score": score})
    scored_sorted = sorted(scored, key=lambda x: x["_ad_score"], reverse=True)
    return scored_sorted[:k]


# ==========================
# 1. СИНТЕТИЧЕСКИЕ ИИ-ПОТРЕБИТЕЛИ
# ==========================

@dataclass
class SyntheticConsumer:
    id: int
    age: int
    interests: List[str]
    behavior: List[str]
    segment: str
    price_sensitivity: float  # 0..1


def generate_synthetic_consumers(n: int = 12) -> List[Dict[str, Any]]:
    """
    Генерирует n профилей "ИИ-потребителей":
    - возраст
    - интересы
    - поведенческие паттерны
    - сегмент (строка для базовой оценки)
    """
    random.seed(42)

    interests_pool = [
        "гаджеты", "игры", "спорт", "музыка", "кино",
        "онлайн-покупки", "скидки", "мода", "умный дом", "путешествия"
    ]
    behavior_pool = [
        "реагирует на скидки",
        "ценит качество",
        "ищет баланс цены и характеристик",
        "любит новинки",
        "доверяет отзывам",
        "берёт по акции"
    ]
    segments = [
        "Low_income_pragmatic_youth",
        "Middle_income_tech_enthusiast",
        "Family_buyer_value_seeker",
        "Premium_quality_oriented"
    ]

    consumers: List[SyntheticConsumer] = []
    for i in range(n):
        age = random.randint(18, 45)
        seg = random.choice(segments)
        interests = random.sample(interests_pool, k=3)
        behavior = random.sample(behavior_pool, k=2)
        price_sensitivity = {
            "Low_income_pragmatic_youth": 0.9,
            "Family_buyer_value_seeker": 0.8,
            "Middle_income_tech_enthusiast": 0.6,
            "Premium_quality_oriented": 0.4,
        }[seg]

        consumers.append(
            SyntheticConsumer(
                id=i + 1,
                age=age,
                interests=interests,
                behavior=behavior,
                segment=seg,
                price_sensitivity=price_sensitivity,
            )
        )

    return [asdict(c) for c in consumers]


# ==========================
# 2. БАЗОВАЯ ОЦЕНКА ОБЪЯВЛЕНИЯ (из твоего старого main.py)
# ==========================

def evaluate_ad(ad_text: str, target_audience: str) -> Dict[str, float]:
    """
    Базовая эвристическая оценка рекламы для одного сегмента.
    Используется как "ядро", поверх которого накладывается поведение потребителей.
    """
    text_lower = ad_text.lower()
    score = 0.5

    if "скид" in text_lower:
        score += 0.20
    if "бесплатн" in text_lower:
        score += 0.10
    if "новин" in text_lower:
        score += 0.05
    if "доставка" in text_lower:
        score += 0.05
    if "хит" in text_lower or "бестселлер" in text_lower:
        score += 0.05

    length = len(ad_text)
    if length < 80:
        score -= 0.10
    elif length > 600:
        score -= 0.10

    if "low_income" in target_audience.lower() and "скид" in text_lower:
        score += 0.05

    click_probability = max(0.0, min(1.0, score))
    purchase_probability = max(0.0, min(1.0, score - 0.1))

    return {
        "click_probability": click_probability,
        "purchase_probability": purchase_probability,
    }


def simulate_ad_for_consumer(
    ad_text: str,
    product: Dict[str, Any],
    consumer: Dict[str, Any]
) -> Dict[str, float]:
    """
    Моделирует реакцию одного ИИ-потребителя на объявление,
    используя базовый скор + поправки на интересы и поведение.
    """
    base_scores = evaluate_ad(ad_text, consumer["segment"])
    score = base_scores["click_probability"]
    purchase = base_scores["purchase_probability"]

    text_lower = ad_text.lower()
    product_text = (
        str(product.get("name", "")) + " " +
        str(product.get("description", "")) + " " +
        str(product.get("category", ""))
    ).lower()

    interests = " ".join(consumer.get("interests", [])).lower()
    behavior = " ".join(consumer.get("behavior", [])).lower()
    price_sens = consumer.get("price_sensitivity", 0.7)

    if any(word in text_lower or word in product_text for word in interests.split()):
        score += 0.05
        purchase += 0.03

    if "реагирует на скидки" in behavior and ("скид" in text_lower or "акция" in text_lower):
        score += 0.08 * price_sens
        purchase += 0.05 * price_sens

    if "любит новинки" in behavior and "новин" in text_lower:
        score += 0.05
        purchase += 0.03

    if "доверяет отзывам" in behavior and any(
        k in text_lower for k in ["выбор покупателей", "отзывы", "рейтинг"]
    ):
        score += 0.04
        purchase += 0.03

    score += random.uniform(-0.02, 0.02)
    purchase += random.uniform(-0.02, 0.02)

    click_probability = max(0.0, min(1.0, score))
    purchase_probability = max(0.0, min(1.0, purchase))

    return {
        "click_probability": click_probability,
        "purchase_probability": purchase_probability,
    }


def evaluate_ad_on_audience(
    ad_text: str,
    product: Dict[str, Any],
    consumers: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Прогоняет объявление по всем ИИ-потребителям и усредняет результат.
    """
    clicks = []
    purchases = []
    for c in consumers:
        scores = simulate_ad_for_consumer(ad_text, product, c)
        clicks.append(scores["click_probability"])
        purchases.append(scores["purchase_probability"])

    return {
        "click_probability": sum(clicks) / len(clicks),
        "purchase_probability": sum(purchases) / len(purchases),
    }


# ==========================
# 3. LLM-КЛИЕНТ ДЛЯ MISTRAL
# ==========================

def get_llm_client(use_mistral: bool = True):
    """
    Возвращает либо реальный MistralClient, либо MockLLMClient.
    В приложении Streamlit по умолчанию используется Mistral.
    """
    if use_mistral:
        return MistralClient()
    return MockLLMClient()


def _advariant_to_dict(v: AdVariant, channel: str) -> Dict[str, str]:
    return {
        "channel": v.channel or channel,
        "headline": v.headline,
        "text": v.text,
        "cta": v.cta,
        "notes": v.notes,
    }


def generate_variants_for_product_channel(
    llm_client,
    product: Dict[str, Any],
    channel: str,
    trends: List[str],
    n_variants: int = 3,
) -> List[Dict[str, str]]:
    """
    Вызывает Mistral (или Mock) для генерации n_variants креативов под один канал.
    """
    payload = {
        "product": {
            "name": product.get("name"),
            "category": product.get("category", "электроника"),
            "price": product.get("price"),
            "margin": product.get("margin"),
            "tags": product.get("tags") or [],
            "features": [
                product.get("description", ""),
                product.get("recommendation", ""),
            ],
        },
        "audience_profile": {
            "age_range": "20-35",
            "interests": ["гаджеты", "онлайн-покупки", "скидки"],
            "behavior": ["реагирует на скидки", "ценит удобство", "ищет выгоду"],
        },
        "channel": channel,
        "trends": trends,
        "n_variants": n_variants,
    }

    variants_objs = llm_client.generate_variants(payload)
    variants: List[Dict[str, str]] = []
    for v in variants_objs:
        if isinstance(v, AdVariant):
            variants.append(_advariant_to_dict(v, channel))
        else:
            variants.append(
                {
                    "channel": v.get("channel", channel),
                    "headline": v.get("headline", ""),
                    "text": v.get("text", ""),
                    "cta": v.get("cta", ""),
                    "notes": v.get("notes", ""),
                }
            )
    return variants


# ==========================
# 4. ПОСТРОЕНИЕ И ОЦЕНКА ОБЪЯВЛЕНИЙ
# ==========================

def build_scored_ads_for_product(
    llm_client,
    product: Dict[str, Any],
    trends: List[str],
    consumers: List[Dict[str, Any]],
    n_variants_per_channel: int = 3,
) -> List[Dict[str, Any]]:
    """
    Для одного товара:
    - генерирует несколько вариантов на каждый из трёх каналов через Mistral
    - прогоняет через ИИ-потребителей
    - возвращает все объявления с оценками
    """
    channels = ["telegram", "vk", "yandex_ads"]
    all_ads: List[Dict[str, Any]] = []

    for ch in channels:
        variants = generate_variants_for_product_channel(
            llm_client=llm_client,
            product=product,
            channel=ch,
            trends=trends,
            n_variants=n_variants_per_channel,
        )

        for v in variants:
            ad_text = f"{v['headline']}\n{v['text']}\n{v['cta']}"
            scores = evaluate_ad_on_audience(ad_text, product, consumers)

            all_ads.append(
                {
                    "product": {
                        "name": product.get("name", ""),
                        "category": product.get("category", ""),
                        "price": product.get("price"),
                    },
                    "channel": ch,
                    "ad": v,
                    "evaluation": scores,
                }
            )

    return all_ads


def pick_best_per_channel(
    scored_ads: List[Dict[str, Any]]
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Для каждой пары (товар, канал) оставляет объявление с максимальной click_probability.
    """
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for item in scored_ads:
        key = (item["product"]["name"], item["channel"])
        current = best.get(key)
        if (
            current is None
            or item["evaluation"]["click_probability"]
            > current["evaluation"]["click_probability"]
        ):
            best[key] = item

    return best


def build_campaign_json(
    best_items: List[Dict[str, Any]],
    consumers: List[Dict[str, Any]],
    niche: str,
    catalog_size: int,
    total_ads_generated: int,
) -> Dict[str, Any]:
    """
    Собирает финальный JSON кампании (совместим с форматом из ТЗ).
    """
    campaigns = []
    for item in best_items:
        campaigns.append(
            {
                "product": item["product"],
                "channel": item["channel"],
                "ad": item["ad"],
                "evaluation": item["evaluation"],
                "targeting": {
                    "audience_segment": "Synthetic_multi_segment",
                    "n_consumers": len(consumers),
                },
                "image_recommendation": {
                    "status": "placeholder",
                    "description": "Рекомендовано светлое фото товара крупным планом на нейтральном фоне.",
                },
            }
        )

    return {
        "platform": "GENAI-4",
        "description": "Полный список сгенерированных и протестированных рекламных креативов по топ-товарам.",
        "niche": niche,
        "n_products_in_catalog": catalog_size,
        "n_top_products_used": len({c["product"]["name"] for c in campaigns}),
        "n_all_ads_generated": total_ads_generated,
        "n_best_ads_in_campaign": len(campaigns),
        "campaigns": campaigns,
    }
