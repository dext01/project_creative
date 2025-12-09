import json
from typing import List, Dict, Any

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

# ==========================
# 0. НАСТРОЙКА СТРАНИЦЫ + CSS
# ==========================

st.set_page_config(
    page_title="GENAI-4 · Автогенерация рекламы",
    layout="wide",
)

st.markdown(
    """
    <style>
    body {
        background-color: #020617;
        color: #e5e7eb;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif;
    }
    .main {
        background: radial-gradient(circle at top left, #020617 0, #0f172a 40%, #020617 100%);
        color: #e5e7eb;
    }
    .section-title {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 6px;
        background: linear-gradient(to right, #e5e7eb, #60a5fa);
        -webkit-background-clip: text;
        color: transparent;
    }
    .section-sub {
        font-size: 13px;
        color: #9ca3af;
        margin-bottom: 18px;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .16em;
        background: rgba(56,189,248,0.1);
        color: #38bdf8;
        border: 1px solid rgba(56,189,248,0.4);
        margin-right: 6px;
    }
    .badge-channel {
        background: rgba(96,165,250,0.15);
        color: #60a5fa;
        border-color: rgba(96,165,250,0.5);
    }
    .top-summary {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 14px 22px;
        border-radius: 999px;
        background: #111827;
        border: 1px solid rgba(148,163,184,0.7);
        margin-bottom: 16px;
        color: #e5e7eb;
        font-size: 14px;
    }
    .top-summary strong {
        color: #f9fafb;
        font-weight: 700;
    }
    .campaign-card {
        border-radius: 20px;
        padding: 18px 20px;
        margin-bottom: 16px;
        background: radial-gradient(circle at top left, #111827 0, #020617 65%);
        box-shadow: 0 18px 40px rgba(15,23,42,0.65);
        border: 1px solid rgba(148,163,184,0.3);
    }
    .headline {
        font-size: 17px;
        font-weight: 650;
        color: #e5e7eb;
        margin-bottom: 4px;
    }
    .product-chip {
        font-size: 12px;
        color: #9ca3af;
        margin-bottom: 8px;
    }
    .cta-chip {
        display: inline-block;
        margin-top: 8px;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(249,115,22,0.16);
        color: #fdba74;
        font-size: 12px;
        border: 1px solid rgba(249,115,22,0.45);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================
# 1. ШАПКА
# ==========================

st.markdown(
    """
    <div style="padding: 8px 0 18px 0;">
      <div style="font-size:13px; letter-spacing:.16em; text-transform:uppercase; color:#6b7280;">
        GENAI-4 · Autonomous Marketing Agent
      </div>
      <div class="section-title">
        Автоматическая генерация и тестирование рекламных объявлений для интернет-магазина
      </div>
      <div class="section-sub">
        Загрузите каталог товаров в JSON/CSV — система выберет лучшие позиции, сгенерирует креативы под Telegram, VK и Yandex Ads,
        прогонит их через синтетическую аудиторию и покажет объявления с наивысшей прогнозируемой конверсией.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================
# 2. САЙДБАР
# ==========================

st.sidebar.header("Входные параметры")

use_real_mistral = st.sidebar.checkbox(
    "Использовать Mistral API (иначе заглушка)",
    value=True,
    help="Для работы нужен ключ MISTRAL_API_KEY в переменных окружения или secrets.",
)

niche = st.sidebar.text_input("Ниша / тип товаров", value="электроника")

trends_input = st.sidebar.text_input(
    "Активные маркетинговые тренды (через запятую)",
    value="минимализм, честность, FOMO, социальное доказательство",
)
trends = [t.strip() for t in trends_input.split(",") if t.strip()]

reruns = st.sidebar.slider(
    "Количество вариантов на канал (n_variants_per_channel)",
    min_value=1,
    max_value=5,
    value=3,
)

uploaded_file = st.sidebar.file_uploader(
    "Каталог товаров (JSON или CSV)",
    type=["json", "csv"],
)

if not uploaded_file:
    st.info("⬅ Загрузите JSON/CSV с каталогом товаров, чтобы сгенерировать кампании.")
    st.stop()

# ==========================
# 3. ЗАГРУЗКА КАТАЛОГА И ВЫБОР ТОВАРОВ
# ==========================

catalog = load_catalog_from_filelike(uploaded_file)
if not catalog:
    st.error("Не удалось прочитать каталог.")
    st.stop()

top_products = select_top_products(catalog, k=3)

# ==========================
# 4. LLM-КЛИЕНТ (MISTRAL ИЛИ MOCK)
# ==========================

try:
    llm_client = get_llm_client(use_mistral=use_real_mistral)
except Exception as e:
    st.error(f"Ошибка инициализации LLM-клиента: {e}")
    st.stop()

# ==========================
# 5. СИНТЕТИЧЕСКАЯ АУДИТОРИЯ
# ==========================

consumers = generate_synthetic_consumers(12)

# ==========================
# 6. ГЕНЕРАЦИЯ И ТЕСТИРОВАНИЕ ОБЪЯВЛЕНИЙ
# ==========================

all_scored_ads: List[Dict[str, Any]] = []

for product in top_products:
    scored_for_product = build_scored_ads_for_product(
        llm_client=llm_client,
        product=product,
        trends=trends,
        consumers=consumers,
        n_variants_per_channel=reruns,
    )
    all_scored_ads.extend(scored_for_product)

if not all_scored_ads:
    st.error("Не удалось сгенерировать объявления (LLM вернул пустой результат).")
    st.stop()

best_per_product_channel = pick_best_per_channel(all_scored_ads)

# 2 лучших объявления по кликабельности — для примера в UI
best_two = sorted(
    all_scored_ads,
    key=lambda x: x["evaluation"]["click_probability"],
    reverse=True,
)[:2]

# ==========================
# 7. ФИНАЛЬНЫЙ JSON КАМПАНИИ
# ==========================

campaign_json = build_campaign_json(
    best_items=list(best_per_product_channel.values()),
    consumers=consumers,
    niche=niche,
    catalog_size=len(catalog),
    total_ads_generated=len(all_scored_ads),
)

# ==========================
# 8. SUMMARY БЛОК
# ==========================

st.markdown(
    f"""
    <div class="top-summary">
      <span class="badge">ГОТОВО</span>
      На основе <strong>{campaign_json['n_products_in_catalog']}</strong> товаров выбрано 
      <strong>{campaign_json['n_top_products_used']}</strong> перспективных позиций. Для них сгенерировано 
      и протестировано <strong>{campaign_json['n_all_ads_generated']}</strong> объявлений
      на синтетической аудитории из <strong>{len(consumers)}</strong> профилей.
      В кампанию вошли лучшие креативы по каждому каналу.
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================
# 9. ПРИМЕРЫ КРЕАТИВОВ
# ==========================

channel_labels = {
    "telegram": "Telegram",
    "vk": "VK",
    "yandex_ads": "Yandex Ads",
}

st.markdown("### ⭐ Примеры креативов (2 объявления с максимальной прогнозируемой кликабельностью)")

for item in best_two:
    p = item["product"]
    a = item["ad"]
    ch_label = channel_labels.get(item["channel"], item["channel"])
    eval_scores = item["evaluation"]

    st.markdown(
        f"""
        <div class="campaign-card">
          <div style="margin-bottom:6px;">
            <span class="badge badge-channel">{ch_label}</span>
            <span class="badge">{p.get('category', 'Без категории')}</span>
          </div>
          <div class="headline">{a['headline']}</div>
          <div class="product-chip">
            Товар: {p.get('name', 'Без названия')} · 
            Примерная цена: {int(p.get('price', 0)) if p.get('price') else '—'} ₽ ·
            Прогноз клика: {(eval_scores['click_probability'] * 100):.1f}% ·
            Прогноз покупки: {(eval_scores['purchase_probability'] * 100):.1f}%
          </div>
          <div style="font-size:13px; color:#d1d5db; margin-bottom:6px;">
            {a['text']}
          </div>
          <div class="cta-chip">CTA: {a['cta']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================
# 10. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ==========================

st.markdown("### 📊 Визуализация результатов тестирования")

viz_rows = []
for item in best_per_product_channel.values():
    viz_rows.append(
        {
            "Товар": item["product"]["name"],
            "Канал": channel_labels.get(item["channel"], item["channel"]),
            "Прогноз клика (%)": item["evaluation"]["click_probability"] * 100,
            "Прогноз покупки (%)": item["evaluation"]["purchase_probability"] * 100,
        }
    )

viz_df = pd.DataFrame(viz_rows)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Кликабельность по товарам и каналам**")
    st.dataframe(viz_df, use_container_width=True)

with col2:
    st.bar_chart(
        viz_df.set_index("Товар")[["Прогноз клика (%)"]],
        use_container_width=True,
    )

# ==========================
# 11. ПОЛНЫЙ JSON + СКАЧИВАНИЕ
# ==========================

st.markdown("### 🧾 Полный JSON со всеми креативами кампании")
st.caption("JSON включает лучшие креативы по каждому товару и каналу, с оценкой клика и покупки, но без генерации картинок (только текст + рекомендация по изображению).")

st.json(campaign_json)

st.download_button(
    label="📥 Скачать JSON кампании",
    file_name="genai4_final_campaign.json",
    mime="application/json",
    data=json.dumps(campaign_json, ensure_ascii=False, indent=4),
)
