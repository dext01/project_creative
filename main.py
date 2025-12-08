# main.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import json
import math
import random

import pandas as pd

from promt import LLMClient, MockLLMClient, AdVariant


# ==========================
# 1. МОДЕЛИ ДАННЫХ
# ==========================

@dataclass
class Product:
    name: str
    category: str
    price: float
    margin: Optional[float] = None
    tags: Optional[List[str]] = None
    description: str = ""


@dataclass
class ConsumerProfile:
    id: str
    age_range: str
    interests: List[str]
    behavior: List[str]
    segment_label: str


@dataclass
class ScoredAd:
    product: Product
    channel: str
    variant: AdVariant
    avg_click_probability: float
    avg_purchase_probability: float


# ==========================
# 2. ЗАГРУЗКА КАТАЛОГА
# ==========================

def load_catalog_from_filelike(file) -> List[Dict[str, Any]]:
    """Streamlit: читает JSON или CSV, возвращает список dict-товаров."""
    name = file.name.lower()
    if name.endswith(".json"):
        data = json.load(file)
        if isinstance(data, dict) and "products" in data:
            return data["products"]
        if isinstance(data, list):
            return data
        raise ValueError("Неожиданный формат JSON: ожидается список или объект с ключом 'products'.")
    elif name.endswith(".csv"):
        df = pd.read_csv(file)
        return df.to_dict(orient="records")
    else:
        raise ValueError("Поддерживаются только JSON и CSV.")


# ==========================
# 3. СКОРИНГ ТОВАРОВ (ТОП-3)
# ==========================

def _safe_float(x, default: float = 0.0) -> float:
    try:
        if isinstance(x, str):
            x = x.replace(" ", "").replace("₽", "").replace(",", ".")
        return float(x)
    except Exception:
        return default


def compute_margin_score(p: Dict[str, Any]) -> float:
    """0..1 по марже."""
    price = _safe_float(p.get("price", 0.0))
    if price <= 0:
        return 0.2

    margin_field = p.get("margin")
    market_cost = _safe_float(p.get("market_cost", 0.0))

    if isinstance(margin_field, (int, float)):
        margin_pct = float(margin_field)
    elif price > 0 and market_cost > 0:
        margin_pct = (price - market_cost) / price * 100
    else:
        margin_pct = 30.0  # дефолт

    score = margin_pct / 80.0
    return max(0.0, min(1.0, score))


def compute_tag_score(p: Dict[str, Any]) -> float:
    tags = p.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    tags_text = " ".join(tags).lower()
    score = 0.0
    if any(k in tags_text for k in ["новинка", "new", "2024", "2025"]):
        score += 0.3
    if any(k in tags_text for k in ["bestseller", "хит", "топ", "hit"]):
        score += 0.3
    if any(k in tags_text for k in ["яркий", "rgb", "подсветка", "стильный"]):
        score += 0.2
    return max(0.0, min(1.0, score))


def compute_visual_score(p: Dict[str, Any]) -> float:
    text = (str(p.get("description", "")) + " " + str(p.get("category", ""))).lower()
    score = 0.0
    if any(k in text for k in ["rgb", "подсветк", "amoled", "oled", "4k", "игров", "геймер"]):
        score += 0.4
    if any(k in text for k in ["компактн", "тонкий", "минимализм"]):
        score += 0.2
    return max(0.0, min(1.0, score))


def compute_product_ad_score(p: Dict[str, Any]) -> float:
    m = compute_margin_score(p)
    t = compute_tag_score(p)
    v = compute_visual_score(p)
    return round(m * 0.5 + t * 0.3 + v * 0.2, 4)


def select_top_products(catalog: List[Dict[str, Any]], k: int = 3) -> List[Product]:
    scored: List[Tuple[Dict[str, Any], float]] = []
    for p in catalog:
        s = compute_product_ad_score(p)
        scored.append((p, s))
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)[:k]

    result: List[Product] = []
    for p, s in scored_sorted:
        result.append(
            Product(
                name=str(p.get("name", "Без названия")),
                category=str(p.get("category", "электроника")),
                price=_safe_float(p.get("price", 0.0)),
                margin=_safe_float(p.get("margin", 0.0)),
                tags=p.get("tags") if isinstance(p.get("tags"), list) else None,
                description=str(p.get("description", "")),
            )
        )
    return result


# ==========================
# 4. СИНТЕТИЧЕСКИЕ ИИ-ПРОФИЛИ
# ==========================

def generate_synthetic_consumers(n: int = 12) -> List[ConsumerProfile]:
    """10+ профилей с разными паттернами поведения."""
    base_profiles = [
        ConsumerProfile(
            id="disc_young",
            age_range="18-24",
            interests=["скидки", "маркетплейсы", "гаджеты"],
            behavior=["реагирует на скидки", "часто покупает онлайн"],
            segment_label="Молодой охотник за скидками",
        ),
        ConsumerProfile(
            id="pragmatic_25_35",
            age_range="25-35",
            interests=["электроника", "работа из дома", "логистика"],
            behavior=["ценит удобную доставку", "сравнивает цены"],
            segment_label="Прагматичный офисный",
        ),
        ConsumerProfile(
            id="eco_lover",
            age_range="25-40",
            interests=["экология", "долговечные вещи"],
            behavior=["читает отзывы", "готов платить за качество"],
            segment_label="Осознанный покупатель",
        ),
        ConsumerProfile(
            id="gamer",
            age_range="18-30",
            interests=["игры", "геймерская периферия", "стримы"],
            behavior=["реагирует на RGB/дизайн", "ценит отзывчивость"],
            segment_label="Геймер",
        ),
        ConsumerProfile(
            id="parent",
            age_range="30-45",
            interests=["товары для дома", "семья"],
            behavior=["ценит надежность", "важна доставка"],
            segment_label="Занятый родитель",
        ),
        ConsumerProfile(
            id="minimalist",
            age_range="20-35",
            interests=["минимализм", "чистый дизайн"],
            behavior=["не любит перегруженный текст"],
            segment_label="Любитель минимализма",
        ),
    ]

    # Если нужно больше n — просто дублируем с небольшим шумом
    result = []
    while len(result) < n:
        for bp in base_profiles:
            if len(result) >= n:
                break
            result.append(bp)
    return result[:n]


# ==========================
# 5. СИМУЛЯЦИЯ ОЦЕНКИ ОБЪЯВЛЕНИЙ
# ==========================

def evaluate_ad_for_profile(text: str, profile: ConsumerProfile) -> Tuple[float, float]:
    """
    Псевдо-оценка вероятности клика/покупки
    на основе кучи эвристик (FOMO, скидки, доставка, дизайн).
    """
    t = text.lower()
    click = 0.03  # базовый CTR
    # скидки
    if "скидк" in t or "распродаж" in t:
        click += 0.07
        if "реагирует на скидки" in profile.behavior:
            click += 0.08
    # FOMO
    if any(k in t for k in ["успей", "пока есть", "количество ограничено"]):
        click += 0.05
    # доставка
    if "доставк" in t and "ценит удобную доставку" in profile.behavior:
        click += 0.05
    # геймеры и RGB
    if any(k in t for k in ["игров", "геймер", "rgb", "подсветк"]):
        if "игры" in profile.interests or "геймерская периферия" in profile.interests:
            click += 0.06
    # минимализм — штраф за «словесный мусор» (условно: “!!!”, куча эмодзи)
    emoji_count = sum(1 for ch in t if ch in "🔥✨💥⭐😍👍👀💡")
    if "минимализм" in profile.interests and emoji_count > 3:
        click -= 0.03

    click = max(0.01, min(0.6, click))
    purchase = click * random.uniform(0.6, 0.9)
    return round(click, 4), round(purchase, 4)


def evaluate_ad_on_audience(
    variant: AdVariant,
    product: Product,
    consumers: List[ConsumerProfile],
) -> Tuple[float, float]:
    text = f"{variant.headline}\n{variant.text}\n{variant.cta}"
    clicks = []
    purchases = []
    for c in consumers:
        c_p, p_p = evaluate_ad_for_profile(text, c)
        clicks.append(c_p)
        purchases.append(p_p)
    avg_click = sum(clicks) / len(clicks)
    avg_purchase = sum(purchases) / len(purchases)
    return round(avg_click, 4), round(avg_purchase, 4)


# ==========================
# 6. ГЕНЕРАЦИЯ КРЕАТИВОВ ДЛЯ ТОВАРА + КАНАЛА
# ==========================

def build_payload_for_llm(
    product: Product,
    channel: str,
    trends: List[str],
    audience_profile: Dict[str, Any],
    n_variants: int = 3,
) -> Dict[str, Any]:
    return {
        "product": {
            "name": product.name,
            "category": product.category,
            "price": product.price,
            "margin": product.margin,
            "tags": product.tags or [],
            "features": [product.description],
        },
        "audience_profile": audience_profile,
        "channel": channel,
        "trends": trends,
        "n_variants": n_variants,
    }


def generate_variants_for_product_channel(
    llm_client,
    product: Product,
    channel: str,
    trends: List[str],
    n_variants: int = 3,
) -> List[AdVariant]:
    audience_profile = {
        "age_range": "20-35",
        "interests": ["электроника", "онлайн-покупки", "скидки"],
        "behavior": ["реагирует на скидки", "ценит удобную доставку"],
    }
    payload = build_payload_for_llm(product, channel, trends, audience_profile, n_variants)
    return llm_client.generate_variants(payload)


def build_image_prompt(product: Product, channel: str, trends: List[str]) -> str:
    """
    Просто текстовое описание для генерации/подбора картинки.
    Это заглушка вместо реального вызова DALL·E.
    """
    trend_text = ", ".join(trends) if trends else "современный минимализм"
    return (
        f"Рекламный баннер для товара '{product.name}' (категория: {product.category}) в стиле '{trend_text}'. "
        f"Чистый фон, акцент на товаре, читаемый текст, формат под канал {channel}."
    )


def build_scored_ads_for_product(
    llm_client,
    product: Product,
    trends: List[str],
    consumers: List[ConsumerProfile],
    n_variants_per_channel: int = 3,
) -> List[ScoredAd]:
    channels = ["telegram", "vk", "yandex_ads"]
    scored_ads: List[ScoredAd] = []

    for ch in channels:
        variants = generate_variants_for_product_channel(
            llm_client=llm_client,
            product=product,
            channel=ch,
            trends=trends,
            n_variants=n_variants_per_channel,
        )
        for v in variants:
            avg_click, avg_purchase = evaluate_ad_on_audience(v, product, consumers)
            scored_ads.append(
                ScoredAd(
                    product=product,
                    channel=ch,
                    variant=v,
                    avg_click_probability=avg_click,
                    avg_purchase_probability=avg_purchase,
                )
            )
    return scored_ads


# ==========================
# 7. ВЫБОР ЛУЧШИХ КРЕАТИВОВ И СБОРКА JSON
# ==========================

def pick_best_per_channel(scored_ads: List[ScoredAd]) -> List[ScoredAd]:
    """
    Для каждого канала берём креатив с максимальным avg_click_probability.
    """
    best_by_channel: Dict[str, ScoredAd] = {}
    for ad in scored_ads:
        ch = ad.channel
        if ch not in best_by_channel:
            best_by_channel[ch] = ad
        else:
            if ad.avg_click_probability > best_by_channel[ch].avg_click_probability:
                best_by_channel[ch] = ad
    return list(best_by_channel.values())


def build_campaign_json(
    niche: str,
    catalog_size: int,
    top_products: List[Product],
    all_scored_ads: List[ScoredAd],
    best_two: List[ScoredAd],
    consumers: List[ConsumerProfile],
) -> Dict[str, Any]:
    consumer_dicts = [
        {
            "id": c.id,
            "age_range": c.age_range,
            "interests": c.interests,
            "behavior": c.behavior,
            "segment_label": c.segment_label,
        }
        for c in consumers
    ]

    # какие именно два примера показываем
    best_ids = {id(ad) for ad in best_two}

    campaigns = []
    for ad in all_scored_ads:
        image_prompt = build_image_prompt(ad.product, ad.channel, trends=[])
        campaigns.append(
            {
                "product": {
                    "name": ad.product.name,
                    "category": ad.product.category,
                    "price": ad.product.price,
                },
                "channel": ad.channel,
                "ad": {
                    "headline": ad.variant.headline,
                    "text": ad.variant.text,
                    "cta": ad.variant.cta,
                    "notes": ad.variant.notes,
                },
                "evaluation": {
                    "click_probability": ad.avg_click_probability,
                    "purchase_probability": ad.avg_purchase_probability,
                },
                "targeting": {
                    "audience_segment": "Synthetic multi-segment",
                    "audience_profiles": consumer_dicts,
                },
                "image_prompt": image_prompt,
                "is_sample_example": (id(ad) in best_ids),
            }
        )

    final_json = {
        "platform": "GENAI-4",
        "description": "Автоматически сгенерированная рекламная кампания по топ-товарам.",
        "niche": niche,
        "n_products_in_catalog": catalog_size,
        "n_top_products_used": len(top_products),
        "n_all_ads": len(campaigns),
        "n_example_ads_shown": len(best_two),
        "campaigns": campaigns,
    }
    return final_json


def get_llm_client():
    """
    Если есть OPENAI_API_KEY — используем реальный LLM.
    Иначе — Mock для оффлайн-демо.
    """
    try:
        return LLMClient()
    except Exception:
        return MockLLMClient()
