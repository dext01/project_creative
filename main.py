# =================================================================
# 1. СТРОКА С __FUTURE__ ДОЛЖНА БЫТЬ ПЕРВОЙ НЕСЛУЖЕБНОЙ СТРОКОЙ!
from __future__ import annotations
# =================================================================

import os
import pandas as pd
import streamlit as st
import json
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# Импорты, которые вы указали:
from sentence_transformers import SentenceTransformer, util
from promt import MistralClient, MockLLMClient, AdVariant


# ==========================
# 2. МОДЕЛИ ДАННЫХ
# ==========================

@dataclass
class Product:
    name: str
    category: str
    price: float
    margin: Optional[float] = None
    tags: Optional[List[str]] = None
    description: str = ""
    # Добавлен атрибут для рекомендации, чтобы использовать его в LLM
    recommendation: Optional[str] = None


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
# 3. ЗАГРУЗКА КАТАЛОГА
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
    elif name.endswith((".csv", ".tsv", ".txt")):
        try:
            df = pd.read_csv(file)
            return df.to_dict(orient="records")
        except Exception as e:
            raise ValueError(f"Ошибка чтения CSV/TSV: {e}")
    else:
        raise ValueError("Поддерживаются только форматы JSON, CSV, TSV.")


# ==========================
# 4. СКОРИНГ ТОВАРОВ (Адаптировано из productAnalyzer.py для синхронной работы)
# ==========================

# Инициализируем SentenceTransformer один раз
@st.cache_resource(show_spinner=False)
def get_semantic_model():
    """Кэшированная загрузка SentenceTransformer."""
    return SentenceTransformer('intfloat/multilingual-e5-base')


# Инициализируем эмбеддинги для скоринга
def get_scoring_embeddings(model):
    """Возвращает кэшированные эмбеддинги для скоринга."""
    return {
        'visual_pos': model.encode(
            ["query: яркий красочный насыщенный неоновый броский дизайн визуально привлекательный"],
            convert_to_tensor=True),
        'visual_neg': model.encode(["query: тусклый серый блеклый простой стандартный обычный скучный матовый"],
                                     convert_to_tensor=True),
        'novelty_pos': model.encode(["query: новинка новый релиз последняя модель 2024 современный инновация тренд"],
                                     convert_to_tensor=True),
        'novelty_neg': model.encode(["query: старый антиквариат устаревший ретро винтаж прошлый век история"],
                                     convert_to_tensor=True),
        'hype_pos': model.encode(["query: бестселлер хит продаж топ популярный выбор покупателей высокий рейтинг"],
                                     convert_to_tensor=True),
        'hype_neg': model.encode(["query: средний неизвестный нишевый базовый запасная часть обыденный"],
                                     convert_to_tensor=True),
    }


MODEL = get_semantic_model()
EMBEDDINGS = get_scoring_embeddings(MODEL)


def _get_semantic_score(embedding: Any, pos: Any, neg: Any) -> float:
    """Вычисляет семантический счет на основе разницы косинусного сходства."""
    # (cos(emb, pos) - cos(emb, neg)) * 100
    score = (util.cos_sim(embedding, pos).item() - util.cos_sim(embedding, neg).item()) * 100
    return max(0, score + 5)


def select_top_products(catalog: List[Dict[str, Any]], k: int = 3) -> List[Product]:
    """
    Выбирает топ-K товаров на основе комплексного скоринга.
    """
    processed = []

    for p in catalog:
        # 1. Семантический скоринг (заменяет старый скоринг по тегам)
        desc_emb = MODEL.encode(f"passage: {p['name']}. {p.get('description', '')}", convert_to_tensor=True)

        m_score = (
                             _get_semantic_score(desc_emb, EMBEDDINGS['visual_pos'], EMBEDDINGS['visual_neg']) +
                             _get_semantic_score(desc_emb, EMBEDDINGS['novelty_pos'], EMBEDDINGS['novelty_neg']) +
                             _get_semantic_score(desc_emb, EMBEDDINGS['hype_pos'], EMBEDDINGS['hype_neg'])
                    ) / 3

        # 2. Маржинальность
        margin = 0
        if p.get('price', 0) > 0 and 'market_cost' in p:
            margin = ((p['price'] - p['market_cost']) / p['price']) * 100

        # 3. Trend (имитация или заглушка, т.к. нет Wordstat API)
        trend_score = random.uniform(5.0, 15.0)

        # Финальный взвешенный счет (аналогично формуле в productAnalyzer.py)
        final_score = (m_score * 1.5) + (margin * 0.4) + trend_score

        processed.append({
            **p,
            "_score_final": final_score,
            "recommendation": (
                f"Скор привлекательности/новизны: {m_score:.1f}. "
                f"Маржинальность: {int(margin)}%. "
                f"Общий скор (демо): {final_score:.1f}."
            )
        })

    top_raw = sorted(processed, key=lambda x: x['_score_final'], reverse=True)[:k]

    # Преобразование в объекты Product
    return [
        Product(
            name=p['name'],
            category=p.get('category', 'N/A'),
            price=p.get('price', 0.0),
            margin=p.get('margin'),
            tags=p.get('tags'),
            description=p.get('description', ''),
            recommendation=p['recommendation'],  # Добавляем результат анализа
        )
        for p in top_raw
    ]


# ==========================
# 5. ГЕНЕРАЦИЯ АУДИТОРИИ (НУЖНО ДОБАВИТЬ КОД, КОТОРЫЙ ВЫ НЕ ПРИСЛАЛИ)
# ==========================

# !!! ВАЖНО: Функции generate_synthetic_consumers и др. должны быть здесь.
# Так как вы не прислали их код, я оставляю заглушки, чтобы избежать ImportError.

def generate_synthetic_consumers() -> List[ConsumerProfile]:
    """Заглушка для функции генерации аудитории."""
    # Замените это на ваш реальный код
    return [
        ConsumerProfile(
            id="C1",
            age_range="25-35",
            interests=["технологии", "спорт"],
            behavior=["часто покупает онлайн"],
            segment_label="Ранний последователь"
        )
    ]

# ... другие функции (evaluate_ad_for_profile, build_scored_ads_for_product, build_campaign_json) ...

# ==========================
# 6. КЛИЕНТ LLM
# ==========================

def get_llm_client():
    """
    Если есть MISTRAL_API_KEY — используем реальный Mistral LLM.
    Иначе — Mock для оффлайн-демо.
    """
    try:
        # Проверяем наличие ключа для Mistral
        if os.getenv("MISTRAL_API_KEY"):
            return MistralClient()
        else:
            raise ValueError("MISTRAL_API_KEY не задан.")
    except Exception as e:
        print(f"Ошибка инициализации MistralClient: {e}. Используется MockLLMClient.")
        return MockLLMClient()


# ==========================
# 7. ДРУГИЕ ФУНКЦИИ (ЗАГЛУШКИ)
# ==========================

# Если у вас есть другие функции, которые импортируются в app.py (например, build_campaign_json),
# вы должны убедиться, что они определены здесь.

def evaluate_ad_for_profile(ad_variant: AdVariant, profile: ConsumerProfile) -> Tuple[float, float]:
    """Заглушка: имитация оценки вероятности клика и покупки."""
    # Замените на вашу реальную логику
    base_click = 0.05
    base_purchase = 0.01

    if "новизна" in ad_variant.ad_text:
        base_click += 0.02
    if profile.age_range == "25-35":
        base_purchase += 0.01

    return base_click, base_purchase

def build_scored_ads_for_product(product: Product) -> List[ScoredAd]:
    """Заглушка: генерирует оцененные варианты объявлений."""
    # Здесь должен быть вызов LLM для генерации вариантов, а затем их оценка.
    # Для целей исправления ошибки импорта, просто возвращаем заглушку.
    mock_variant = AdVariant(
        title="Демо-Заголовок",
        body="Демо-Текст объявления.",
        keywords=["демо", product.name]
    )
    click, purchase = evaluate_ad_for_profile(mock_variant, generate_synthetic_consumers()[0])

    return [
        ScoredAd(
            product=product,
            channel="social_media",
            variant=mock_variant,
            avg_click_probability=click,
            avg_purchase_probability=purchase
        )
    ]

def pick_best_per_channel(scored_ads: List[ScoredAd]) -> Dict[str, ScoredAd]:
    """Заглушка: выбирает лучшее объявление для каждого канала."""
    best = {}
    for ad in scored_ads:
        if ad.channel not in best or ad.avg_purchase_probability > best[ad.channel].avg_purchase_probability:
            best[ad.channel] = ad
    return best

def build_campaign_json(best_ads: Dict[str, ScoredAd]) -> Dict[str, Any]:
    """Заглушка: создает финальный JSON кампании."""
    return {
        "campaign_name": "Demo_Campaign",
        "best_ads": {
            channel: {
                "product_name": ad.product.name,
                "ad_title": ad.variant.title,
                "ad_body": ad.variant.body,
                "score": f"{ad.avg_purchase_probability:.4f}"
            }
            for channel, ad in best_ads.items()
        }
    }
