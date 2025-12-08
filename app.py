import json
from typing import List

import streamlit as st
import pandas as pd

from main import (
    load_catalog_from_filelike,
    select_top_products,
    generate_synthetic_consumers,
    build_scored_ads_for_product,
    pick_best_per_channel,
    build_campaign_json,
    get_llm_client,
)


st.set_page_config(
    page_title="GENAI-4 · Авто-реклама для интернет-магазина",
    layout="wide",
)


# ==========================
# CSS (без жёстких трюков, чтобы всё показывалось)
# ==========================

st.markdown(
    """
    <style>
    body {
        background-color: #020617;
        color: #e5e7eb;
        font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    }
    .main {
        background: radial-gradient(circle at top left, #020617 0, #0b1120 40%, #020617 100%);
        color: #e5e7eb;
    }
    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin: 12px 0 4px 0;
    }
    .section-sub {
        font-size: 13px;
        color: #9ca3af;
        margin-bottom: 12px;
    }
    .tag-pill {
        display:inline-block;
        padding:2px 10px;
        border-radius:999px;
        border:1px solid #4b5563;
        font-size:11px;
        margin-right:6px;
        color:#e5e7eb;
    }
    .card {
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 12px;
        background: #020617;
        border: 1px solid #1f2937;
    }
    .headline {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .meta {
        font-size: 12px;
        color: #9ca3af;
        margin-bottom: 6px;
    }
    .cta-chip {
        display:inline-block;
        margin-top:8px;
        padding:4px 10px;
        border-radius:999px;
        border:1px solid rgba(129, 230, 217, 0.5);
        background:rgba(20, 184, 166, 0.1);
        color:#5eead4;
        font-size:12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================
# SIDEBAR
# ==========================

with st.sidebar:
    st.header("⚙️ Настройки кампании")
    niche = st.text_input("Ниша магазина", value="Интернет-магазин электроники")
    trends_text = st.text_input(
        "Маркетинговые тренды (через запятую)",
        value="минимализм, честность, FOMO, социальное доказательство",
    )
    trends: List[str] = [t.strip() for t in trends_text.split(",") if t.strip()]

    uploaded_file = st.file_uploader(
        "Каталог товаров (JSON или CSV)",
        type=["json", "csv"],
        help="Минимум: поля name, category, price, description, margin/tags — по возможности.",
    )

    st.markdown("---")
    st.caption(
        "Если OPENAI_API_KEY не задан, тексты будут сгенерированы простым шаблоном (Mock), "
        "но вся логика анализа и симуляции сохранится."
    )


# ==========================
# HEADER
# ==========================

st.markdown(
    """
    <div>
      <div style="font-size:13px; text-transform:uppercase; letter-spacing:.18em; color:#6b7280;">
        GENAI-4 · Autonomous Marketing Agent
      </div>
      <div class="section-title">
        Автоматическая генерация и тестирование рекламных креативов
      </div>
      <div class="section-sub">
        Загрузите каталог товаров — система выберет лучшие позиции, сгенерирует креативы под Telegram, VK и Yandex Ads,
        протестирует их на синтетической аудитории и соберёт JSON-кампанию для запуска.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if not uploaded_file:
    st.info("⬅ Загрузите каталог товаров в сайдбаре, чтобы запустить генерацию кампании.")
    st.stop()

# ==========================
# 1. ЗАГРУЗКА КАТАЛОГА
# ==========================

try:
    raw_catalog = load_catalog_from_filelike(uploaded_file)
except Exception as e:
    st.error(f"Ошибка при загрузке каталога: {e}")
    st.stop()

if not raw_catalog:
    st.error("Каталог пустой или не удалось его прочитать.")
    st.stop()

st.subheader("1. Каталог загружен")
st.write(f"Найдено товаров в каталоге: **{len(raw_catalog)}**")
st.dataframe(pd.DataFrame(raw_catalog).head(10))


# ==========================
# 2. ВЫБОР ТОП-3 ТОВАРОВ
# ==========================

top_products = select_top_products(raw_catalog, k=3)
st.subheader("2. Топ-товары для рекламы (по марже, тегам и визуальности)")

df_top = pd.DataFrame(
    [
        {
            "Название": p.name,
            "Категория": p.category,
            "Цена": p.price,
            "Маржа (приблизительно)": p.margin,
            "Теги": ", ".join(p.tags or []),
        }
        for p in top_products
    ]
)
st.dataframe(df_top)


# ==========================
# 3. СИНТЕТИЧЕСКАЯ АУДИТОРИЯ
# ==========================

st.subheader("3. Синтетическая целевая аудитория (ИИ-профили)")

consumers = generate_synthetic_consumers(n=12)
df_consumers = pd.DataFrame(
    [
        {
            "ID": c.id,
            "Возраст": c.age_range,
            "Сегмент": c.segment_label,
            "Интересы": ", ".join(c.interests),
            "Поведение": ", ".join(c.behavior),
        }
        for c in consumers
    ]
)
st.dataframe(df_consumers)


# ==========================
# 4. ГЕНЕРАЦИЯ И ТЕСТИРОВАНИЕ КРЕАТИВОВ
# ==========================

st.subheader("4. Генерация креативов и симуляция отклика")

llm_client = get_llm_client()

all_scored_ads = []
for p in top_products:
    scored_for_product = build_scored_ads_for_product(
        llm_client=llm_client,
        product=p,
        trends=trends,
        consumers=consumers,
        n_variants_per_channel=3,
    )
    # для каждого продукта возьмём лучший по каждому каналу
    best_per_channel = pick_best_per_channel(scored_for_product)
    all_scored_ads.extend(best_per_channel)

# ===== табличка с результатами =====
results_rows = []
for ad in all_scored_ads:
    results_rows.append(
        {
            "Товар": ad.product.name,
            "Канал": ad.channel,
            "CTR (симуляция)": ad.avg_click_probability,
            "Конверсия в покупку": ad.avg_purchase_probability,
        }
    )
df_results = pd.DataFrame(results_rows)
st.write("Сводка по лучшим креативам для каждого товара и канала:")
st.dataframe(df_results)


# ==========================
# 5. ВЫБОР 2 ЛУЧШИХ КРЕАТИВОВ И ВИЗУАЛИЗАЦИЯ
# ==========================

st.subheader("5. Примеры креативов (2 лучших по прогнозируемой конверсии)")

# сортируем по CTR и берём top-2
sorted_ads = sorted(
    all_scored_ads,
    key=lambda x: x.avg_click_probability,
    reverse=True,
)
best_two = sorted_ads[:2]

channel_label = {
    "telegram": "Telegram",
    "vk": "VK",
    "yandex_ads": "Yandex Ads",
}

for ad in best_two:
    st.markdown(
        f"""
        <div class="card">
          <div class="meta">
            <span class="tag-pill">{channel_label.get(ad.channel, ad.channel)}</span>
            <span class="tag-pill">{ad.product.category}</span>
          </div>
          <div class="headline">{ad.variant.headline}</div>
          <div class="meta">Товар: {ad.product.name} · Примерная цена: {int(ad.product.price) if ad.product.price else "-"} ₽</div>
          <div style="font-size:13px; margin-bottom:6px;">{ad.variant.text}</div>
          <div class="cta-chip">CTA: {ad.variant.cta}</div>
          <div style="font-size:11px; color:#9ca3af; margin-top:6px;">
            CTR (симуляция): {ad.avg_click_probability:.3f} · Конверсия в покупку: {ad.avg_purchase_probability:.3f}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================
# 6. ФИНАЛЬНЫЙ JSON + СКАЧИВАНИЕ
# ==========================

st.subheader("6. Финальный JSON кампании")

final_json = build_campaign_json(
    niche=niche,
    catalog_size=len(raw_catalog),
    top_products=top_products,
    all_scored_ads=all_scored_ads,
    best_two=best_two,
    consumers=consumers,
)

st.caption(
    "Это итоговая структура рекламной кампании. Она содержит все креативы, "
    "симулированные метрики и профили аудитории. Поле `is_sample_example=true` — "
    "два объявления, показанные выше."
)

st.download_button(
    label="📥 Скачать JSON со всеми объявлениями",
    file_name="genai4_campaign.json",
    mime="application/json",
    data=json.dumps(final_json, ensure_ascii=False, indent=2),
)

with st.expander("Показать JSON здесь"):
    st.json(final_json)
