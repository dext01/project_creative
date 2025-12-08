import json
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
import random


# ==========================
# 0. НАСТРОЙКА СТРАНИЦЫ + CSS
# ==========================

st.set_page_config(
    page_title="GENAI-4 · Автогенерация рекламы",
    layout="wide"
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
        letter-spacing: .08em;
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
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================

def load_catalog(file) -> List[Dict[str, Any]]:
    """Загрузка каталога из JSON или CSV в список словарей."""
    name = file.name.lower()
    if name.endswith(".json"):
        data = json.load(file)
        if isinstance(data, dict) and "products" in data:
            data = data["products"]
        return data
    else:
        df = pd.read_csv(file)
        return df.to_dict(orient="records")


def compute_margin_score(product: Dict[str, Any]) -> float:
    price = float(product.get("price", 0) or 0)
    market_cost = product.get("market_cost")
    margin_field = product.get("margin")

    if isinstance(margin_field, (int, float)):
        margin_percent = float(margin_field)
    elif price > 0 and market_cost is not None:
        margin_percent = (price - float(market_cost)) / price * 100
    else:
        margin_percent = 30.0  # дефолт

    return max(0.0, min(1.0, margin_percent / 80.0))


def compute_tag_score(product: Dict[str, Any]) -> float:
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


def compute_visual_score(product: Dict[str, Any]) -> float:
    desc = str(product.get("description") or "") + " " + str(product.get("category") or "")
    text = desc.lower()
    score = 0.0
    if any(k in text for k in ["rgb", "подсветка", "amoled", "красив", "дизайн"]):
        score += 0.4
    if any(k in text for k in ["компакт", "минимализм", "тонкий"]):
        score += 0.2
    return max(0.0, min(1.0, score))


def compute_product_ad_score(product: Dict[str, Any]) -> float:
    m = compute_margin_score(product)
    t = compute_tag_score(product)
    v = compute_visual_score(product)
    return round((m * 0.5 + t * 0.3 + v * 0.2), 3)


def select_top_products(catalog: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    scored = []
    for p in catalog:
        score = compute_product_ad_score(p)
        scored.append({**p, "_ad_score": score})
    scored_sorted = sorted(scored, key=lambda x: x["_ad_score"], reverse=True)
    return scored_sorted[:k]


def build_trend_phrase(trends: List[str]) -> str:
    if not trends:
        return ""
    return ", ".join(trends)


# ==========================
# 2. ГЕНЕРАЦИЯ ТЕКСТОВ ДЛЯ КАНАЛОВ
# ==========================

def generate_telegram_variants(product: Dict[str, Any], trends: List[str], n_variants: int = 1) -> List[Dict[str, str]]:
    """Телега: коротко, эмоционально."""
    name = product.get("name", "товар")
    desc = product.get("description", "")
    price = product.get("price")
    tags = product.get("tags") or []
    tags_text = ", ".join(tags) if isinstance(tags, list) else str(tags)

    variants = []
    for i in range(n_variants):
        headline = random.choice([
            f"{name} — забери, пока есть",
            f"{name}: новинка для тебя",
            f"{name} — техника без лишнего шума"
        ])

        base = f"{name} — {desc}" if desc else name
        text_parts = [base]

        if price:
            text_parts.append(f"Сейчас около {int(price)} ₽.")
        if "новинка" in tags_text.lower():
            text_parts.append("Свежий релиз, пока мало у кого есть.")
        if any(k in tags_text.lower() for k in ["bestseller", "хит", "hit"]):
            text_parts.append("Уже стал хитом у покупателей.")
        text_parts.append(random.choice([
            "Успей, пока цена ещё держится 🔥",
            "Пока есть в наличии — лучшее время забрать.",
            "Количество ограничено, не откладывай."
        ]))

        text = " ".join(text_parts)
        cta = random.choice(["Успеть взять сейчас", "Перейти к покупке"])

        variants.append({
            "channel": "telegram",
            "headline": headline,
            "text": text,
            "cta": cta,
            "notes": f"TG, {build_trend_phrase(trends)}"
        })
    return variants


def generate_vk_variants(product: Dict[str, Any], trends: List[str], n_variants: int = 1) -> List[Dict[str, str]]:
    """VK: больше текста, соцдоказательство."""
    name = product.get("name", "товар")
    desc = product.get("description", "")
    price = product.get("price")
    tags = product.get("tags") or []
    tags_text = ", ".join(tags) if isinstance(tags, list) else str(tags)

    variants = []
    for _ in range(n_variants):
        headline = random.choice([
            f"{name}: техника, которая радует каждый день",
            f"{name} — выбор тех, кто ценит качество",
            f"{name} для дома и работы"
        ])

        text = f"{name} — для тех, кто хочет получать максимум от техники. {desc} "
        if price:
            text += f"Сейчас доступно примерно за {int(price)} ₽. "
        if "bestseller" in tags_text.lower():
            text += "Один из самых популярных товаров у наших клиентов. "
        if "новинка" in tags_text.lower():
            text += "Новая модель, которая только появилась в продаже. "
        text += "Оформляйте заказ онлайн — доставка и гарантия включены."

        variants.append({
            "channel": "vk",
            "headline": headline,
            "text": text,
            "cta": "Заказать онлайн",
            "notes": f"VK, {build_trend_phrase(trends)}, соцдоказательство"
        })
    return variants


def generate_yandex_variants(product: Dict[str, Any], trends: List[str], n_variants: int = 1) -> List[Dict[str, str]]:
    """Yandex Ads: строго, коротко, без эмодзи."""
    name = product.get("name", "товар")
    desc = product.get("description", "")
    price = product.get("price")

    variants = []
    for _ in range(n_variants):
        headline = random.choice([
            f"{name} со скидкой",
            f"{name} — выгодная цена",
            f"{name} с быстрой доставкой"
        ])

        text = desc or ""
        if price:
            if text:
                text += " "
            text += f"Цена около {int(price)} ₽. Быстрая доставка."

        variants.append({
            "channel": "yandex_ads",
            "headline": headline,
            "text": text.strip(),
            "cta": "Купить онлайн",
            "notes": f"Yandex Ads, {build_trend_phrase(trends)}, ключевые выгоды"
        })
    return variants


# ==========================
# 3. ВНУТРЕННИЙ СКОР ОБЪЯВЛЕНИЙ
# ==========================

def score_ad_variant(ad: Dict[str, str], product: Dict[str, Any]) -> float:
    """
    Внутренний скор качества объявления: чем выше, тем "лучше".
    Используется ТОЛЬКО для выбора лучших вариантов.
    """
    text_all = (ad["headline"] + " " + ad["text"]).lower()

    score = product.get("_ad_score", 0.5)  # базово — насколько товар хорош для рекламы

    # FOMO
    if any(k in text_all for k in ["успей", "пока есть", "только сегодня", "акция", "ограничено"]):
        score += 0.15

    # скидки
    if "скид" in text_all or "со скидкой" in text_all:
        score += 0.12

    # новинка / хит
    if any(k in text_all for k in ["новинка", "новая модель", "свежий релиз"]):
        score += 0.08
    if any(k in text_all for k in ["хит продаж", "бестселлер", "выбор покупателей", "популярный товар"]):
        score += 0.08

    # канал
    if ad["channel"] == "telegram":
        score += 0.03
    if ad["channel"] == "yandex_ads":
        score += 0.04  # чуть выше, как перфоманс-канал

    # маленький рандом, чтобы не было полного равенства
    score += random.uniform(-0.01, 0.01)

    return round(max(0.0, min(1.0, score)), 3)


# ==========================
# 4. ВЫБОР ЛУЧШЕГО ОБЪЯВЛЕНИЯ ДЛЯ КАЖДОГО КАНАЛА
# ==========================

def generate_best_for_channel(product: Dict[str, Any],
                              trends: List[str],
                              channel: str,
                              reruns: int = 5) -> Dict[str, str]:
    """
    Для заданного товара и канала:
    - генерируем reruns вариантов
    - считаем скор
    - возвращаем один лучший вариант
    """
    if channel == "telegram":
        generator = generate_telegram_variants
    elif channel == "vk":
        generator = generate_vk_variants
    elif channel == "yandex_ads":
        generator = generate_yandex_variants
    else:
        raise ValueError(f"Неизвестный канал: {channel}")

    best_variant = None
    best_score = -1.0

    for _ in range(reruns):
        variant = generator(product, trends, n_variants=1)[0]
        s = score_ad_variant(variant, product)
        if s > best_score:
            best_score = s
            best_variant = {**variant}  # копия

    best_variant["_internal_score"] = best_score
    return best_variant


def generate_best_variants_for_product(product: Dict[str, Any],
                                       trends: List[str],
                                       reruns: int = 5) -> List[Dict[str, Any]]:
    """
    Для одного товара:
    - Telegram: лучший из reruns
    - VK: лучший из reruns
    - Yandex Ads: лучший из reruns
    => 3 объявления на 1 товар
    """
    best_tg = generate_best_for_channel(product, trends, "telegram", reruns)
    best_vk = generate_best_for_channel(product, trends, "vk", reruns)
    best_ya = generate_best_for_channel(product, trends, "yandex_ads", reruns)
    return [best_tg, best_vk, best_ya]


# ==========================
# 5. UI
# ==========================

# --- шапка ---
st.markdown(
    """
    <div style="padding: 8px 0 18px 0;">
      <div style="font-size:13px; letter-spacing:.16em; text-transform:uppercase; color:#6b7280;">
        GENAI-4 · Autonomous Marketing Agent
      </div>
      <div class="section-title">
        Автоматическая генерация рекламных объявлений для интернет-магазина
      </div>
      <div class="section-sub">
        Загрузите каталог товаров в JSON/CSV — система выберет 3 лучших товара и
        сгенерирует креативы под Telegram, VK и Yandex Ads. Для каждого товара
        для каждого канала генерируется по несколько вариантов, выбирается лучший.
        Ниже показаны только два самых сильных примера, а полный набор доступен в JSON.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- сайдбар ---
st.sidebar.header("Входные параметры")

niche = st.sidebar.text_input("Ниша / тип товаров", value="электроника")
trends_input = st.sidebar.text_input(
    "Активные маркетинговые тренды (через запятую)",
    value="минимализм, честность, FOMO, социальное доказательство"
)
trends = [t.strip() for t in trends_input.split(",") if t.strip()]

uploaded_file = st.sidebar.file_uploader(
    "Каталог товаров (JSON или CSV)",
    type=["json", "csv"]
)

if not uploaded_file:
    st.info("⬅ Загрузите JSON/CSV с каталогом товаров, чтобы сгенерировать кампании.")
    st.stop()

# --- загрузка и анализ ---
catalog = load_catalog(uploaded_file)
if not catalog:
    st.error("Не удалось прочитать каталог.")
    st.stop()

top_products = select_top_products(catalog, k=3)

# генерируем ЛУЧШИЕ объявления для топ-товаров:
# 3 товара × 3 канала = 9 объявлений
all_variants: List[Dict[str, Any]] = []
for product in top_products:
    best_3_for_product = generate_best_variants_for_product(product, trends, reruns=5)
    for v in best_3_for_product:
        all_variants.append({
            "product": product,
            "ad": v,
            "score": v.get("_internal_score", 0.0)
        })

# выбираем 2 лучших по внутреннему скору — для ПРИМЕРА
best_two = sorted(all_variants, key=lambda x: x["score"], reverse=True)[:2]
best_ids = {id(x) for x in best_two}

# готовим JSON со ВСЕМИ 9 объявлениями
campaigns_all = []
for item in all_variants:
    p = item["product"]
    a = item["ad"]
    is_sample = id(item) in best_ids
    campaigns_all.append({
        "product": {
            "name": p.get("name", ""),
            "category": p.get("category", ""),
            "price": p.get("price", None),
        },
        "channel": a["channel"],
        "ad": {
            "headline": a["headline"],
            "text": a["text"],
            "cta": a["cta"],
            "notes": a["notes"],
        },
        "internal_score": item["score"],     # служебное поле
        "is_sample_example": is_sample       # True для двух показанных в UI
    })

final_json = {
    "platform": "GENAI-4",
    "description": "Сгенерированные рекламные креативы по топ-товарам (3 товара × 3 канала = 9 объявлений).",
    "niche": niche,
    "n_products_in_catalog": len(catalog),
    "n_top_products_used": len(top_products),
    "n_all_ads": len(campaigns_all),
    "n_example_ads_shown": len(best_two),
    "campaigns": campaigns_all
}

# --- summary ---
st.markdown(
    f"""
    <div class="top-summary">
      <span class="badge">ГОТОВО</span>
      На основе <strong>{final_json['n_products_in_catalog']}</strong> товаров выбрано 
      <strong>{final_json['n_top_products_used']}</strong> лучших позиций и сгенерировано 
      <strong>{final_json['n_all_ads']}</strong> объявлений (3 товара × 3 канала).
      Ниже показаны только два примера, полный набор доступен в JSON.
    </div>
    """,
    unsafe_allow_html=True,
)

# --- карточки двух лучших объявлений (пример креативов) ---
channel_labels = {
    "telegram": "Telegram",
    "vk": "VK",
    "yandex_ads": "Yandex Ads"
}

st.markdown(f"### ⭐ Примеры креативов (2 из {final_json['n_all_ads']})")

for item in best_two:
    p = item["product"]
    a = item["ad"]
    ch_label = channel_labels.get(a["channel"], a["channel"])

    st.markdown(
        f"""
        <div class="campaign-card">
          <div style="margin-bottom:6px;">
            <span class="badge badge-channel">{ch_label}</span>
            <span class="badge">{p.get('category', 'Без категории')}</span>
          </div>
          <div class="headline">{a['headline']}</div>
          <div class="product-chip">
            Товар: {p.get('name', 'Без названия')} · Примерная цена: {int(p.get('price', 0)) if p.get('price') else '—'} ₽
          </div>
          <div style="font-size:13px; color:#d1d5db;">
            {a['text']}
          </div>
          <div class="cta-chip">CTA: {a['cta']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- JSON + скачивание ВСЕХ 9 реклам ---
st.markdown("### 🧾 Полный JSON со всеми 9 сгенерированными объявлениями")
st.caption(
    "JSON включает по одному лучшему объявлению на каждый канал для каждого из трёх топ-товаров. "
    "Поле `is_sample_example=true` отмечает два объявления, показанные выше как пример."
)

st.json(final_json)

st.download_button(
    label="📥 Скачать JSON со всеми креативами",
    file_name="genai4_all_ads.json",
    mime="application/json",
    data=json.dumps(final_json, ensure_ascii=False, indent=4),
)
