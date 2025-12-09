import json
import asyncio
import httpx
import math
import os
from sentence_transformers import SentenceTransformer, util


class ProductAnalyzer:
    def __init__(self, JSON_FILE):
        self.model = SentenceTransformer('intfloat/multilingual-e5-base')

        self.visual_pos = self.model.encode(
            ["query: яркий красочный насыщенный неоновый броский дизайн визуально привлекательный"],
            convert_to_tensor=True
        )
        self.visual_neg = self.model.encode(
            ["query: тусклый серый блеклый простой стандартный обычный скучный матовый"],
            convert_to_tensor=True
        )

        self.novelty_pos = self.model.encode(
            ["query: новинка новый релиз последняя модель 2024 современный инновация тренд"],
            convert_to_tensor=True
        )
        self.novelty_neg = self.model.encode(
            ["query: старый антиквариат устаревший ретро винтаж прошлый век история"],
            convert_to_tensor=True
        )

        self.hype_pos = self.model.encode(
            ["query: бестселлер хит продаж топ популярный выбор покупателей высокий рейтинг"],
            convert_to_tensor=True
        )
        self.hype_neg = self.model.encode(
            ["query: средний неизвестный нишевый базовый запасная часть обыденный"],
            convert_to_tensor=True
        )

        self.OAUTH_TOKEN = os.getenv("OAUTH_TOKEN")
        self.JSON_FILE = JSON_FILE

    def _get_score(self, embedding, pos, neg):
        score = (util.cos_sim(embedding, pos).item() - util.cos_sim(embedding, neg).item()) * 100
        return max(0, score + 5)

    async def get_trend_info(self, phrase_name):
        if not self.OAUTH_TOKEN:
            return None

        url = "https://api.wordstat.yandex.net/v1/topRequests"
        payload = {"phrase": phrase_name, "devices": ["phone", "desktop"]}
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self.OAUTH_TOKEN}"
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except Exception:
                return None

    async def run(self):
        with open(self.JSON_FILE, 'r', encoding='utf-8') as f:
            products = json.load(f)

        tasks = [self.get_trend_info(p['name']) for p in products]
        api_responses = await asyncio.gather(*tasks)

        processed = []

        for i, p in enumerate(products):
            json_data = api_responses[i]
            total_trend = 0

            if json_data and 'topRequests' in json_data:
                for item in json_data['topRequests']:
                    total_trend += item.get('count', 0)

            desc_emb = self.model.encode(
                f"passage: {p['name']}. {p['description']}",
                convert_to_tensor=True
            )

            m_score = (
                self._get_score(desc_emb, self.visual_pos, self.visual_neg)
                + self._get_score(desc_emb, self.novelty_pos, self.novelty_neg)
                + self._get_score(desc_emb, self.hype_pos, self.hype_neg)
            ) / 3

            margin = 0
            if p['price'] > 0:
                margin = ((p['price'] - p['market_cost']) / p['price']) * 100

            trend_score = math.log1p(total_trend) * 2.5
            final = (m_score * 1.5) + (margin * 0.4) + trend_score

            processed.append({**p, "_temp_final": final, "_trend": total_trend, "_margin": margin})

        top3 = sorted(processed, key=lambda x: x['_temp_final'], reverse=True)[:3]

        final_output = []
        for item in top3:
            clean = {k: v for k, v in item.items() if not k.startswith('_')}
            clean["recommendation"] = (
                f"Спрос: {item['_trend']}, "
                f"Маржинальность: {int(item['_margin'])}%, "
                f"Итоговый скор: {item['_temp_final']:.1f}"
            )
            final_output.append(clean)

        with open("best_products.json", "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(ProductAnalyzer("products.json").run())
