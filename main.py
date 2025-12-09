from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import json
import math
import random
import os  # Обязательно для чтения переменных окружения

import pandas as pd
import streamlit as st  # <--- ДОБАВЛЕНО/ИСПРАВЛЕНО
from sentence_transformers import SentenceTransformer, util # <--- ДОБАВЛЕНО/ИСПРАВЛЕНО

# Убедитесь, что вы импортируете MistralClient, а не старый LLMClient
from promt import MistralClient, MockLLMClient, AdVariant

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
    elif name.endswith((".csv", ".tsv", ".txt")):
        try:
            df = pd.read_csv(file)
            return df.to_dict(orient="records")
        except Exception as e:
            raise ValueError(f"Ошибка чтения CSV/TSV: {e}")
    else:
        raise ValueError("Поддерживаются только форматы JSON, CSV, TSV.")


# ==========================
# 3. СКОРИНГ ТОВАРОВ (Адаптировано из productAnalyzer.py для синхронной работы)
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
    ВНИМАНИЕ: Здесь отсутствует вызов Yandex API, т.к. Streamlit синхронен.
    Скор тренда имитируется/отсутствует.
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
        # Если в товаре уже есть поле '_temp_trend' (например, если он был предварительно проанализирован)
        # ИЛИ просто используем константу/случайное число для имитации тренда в Streamlit-демо
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
# 4. ГЕНЕРАЦИЯ АУДИТОРИИ
# ==========================

# ... (Остальная часть main.py остается прежней, включая generate_synthetic_consumers, evaluate_ad_for_profile и т.д.)

# ==========================
# 5. КЛИЕНТ LLM
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

# (Оставшаяся часть main.py, включая build_scored_ads_for_product, pick_best_per_channel, build_campaign_json,
# остается без изменений, кроме использования MistralClient вместо LLMClient в get_llm_client)
