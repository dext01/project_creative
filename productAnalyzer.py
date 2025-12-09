import json
import asyncio
import httpx
import math
import os
import random
from typing import Dict, Any, List, Optional

from sentence_transformers import SentenceTransformer, util


class ProductAnalyzer:
    """
    Анализирует каталог товаров, используя семантические эмбеддинги (SentenceTransformer)
    и асинхронно запрашивая тренды через API Wordstat (или заглушку, если токена нет).
    """

    def __init__(self, JSON_FILE: str):
        # Инициализация модели эмбеддингов
        self.model = SentenceTransformer("intfloat/multilingual-e5-base")

        # Эмбеддинги для семантического скоринга
        self.visual_pos = self.model.encode(
            ["query: яркий красочный насыщенный неоновый броский дизайн визуально привлекательный"],
            convert_to_tensor=True,
        )
        self.visual_neg = self.model.encode(
            ["query: тусклый серый блеклый простой стандартный обычный скучный матовый"],
            convert_to_tensor=True,
        )

        self.novelty_pos = self.model.encode(
            ["query: новинка новый релиз последняя модель 2024 современный инновация тренд"],
            convert_to_tensor=True,
        )
        self.novelty_neg = self.model.encode(
            ["query: старый антиквариат устаревший ретро винтаж прошлый век история"],
            convert_to_tensor=True,
        )

        self.hype_pos = self.model.encode(
            ["query: бестселлер хит продаж топ популярный выбор покупателей высокий рейтинг"],
            convert_to_tensor=True,
        )
        self.hype_neg = self.model.encode(
            ["query: средний неизвестный нишевый базовый запасная часть обыденный"],
            convert_to_tensor=True,
        )

        # Токен Wordstat
        self.OAUTH_TOKEN = os.getenv("YANDEX_OAUTH_TOKEN")

        self.JSON_FILE = JSON_FILE

    def _get_score(self, embedding: Any, pos: Any, neg: Any) -> float:
        """Вычисляет семантический счет на основе разницы косинусного сходства."""
        score = (util.cos_sim(embedding, pos).item() - util.cos_sim(embedding, neg).item()) * 100
        return max(0.0, score + 5.0)

    async def get_trend_info(self, phrase_name: str) -> Optional[Dict[str, Any]]:
        """
        Асинхронный запрос к Wordstat.
        Если токен не задан — возвращаем заглушку.
        """
        if not self.OAUTH_TOKEN:
            print("ВНИМАНИЕ: YANDEX_OAUTH_TOKEN не задан. Используется заглушка для трендов.")
            await asyncio.sleep(0.1)
            return {"topRequests": [{"count": random.choice([80, 200, 1000])}]}

        url = "https://api.wordstat.yandex.net/v1/topRequests"

        payload = {
            "phrase": phrase_name,
            "devices": ["phone", "desktop"],
        }

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self.OAUTH_TOKEN}",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                print(f"HTTP ошибка для '{phrase_name}': {e}")
                return None
            except Exception as e:
                print(f"Ошибка соединения для '{phrase_name}': {e}")
                return None

    async def run(self):
        """Запускает полный анализ товаров и сохраняет топ-3 в best_products.json."""
        try:
            with open(self.JSON_FILE, "r", encoding="utf-8") as f:
                products = json.load(f)
        except FileNotFoundError:
            print(f"Файл {self.JSON_FILE} не найден.")
            return

        tasks = [self.get_trend_info(p["name"]) for p in products]
        api_responses = await asyncio.gather(*tasks)

        processed: List[Dict[str, Any]] = []

        print(f"\n{'ТОВАР':<25} | {'СПРОС (Сумма)':<13} | {'СЧЁТ'}")
        print("-" * 55)

        for i, p in enumerate(products):
            json_data = api_responses[i]
            total_trend = 0
            if json_data and "topRequests" in json_data:
                for item in json_data["topRequests"]:
                    total_trend += item.get("count", 0)

            desc_emb = self.model.encode(
                f"passage: {p['name']}. {p.get('description', '')}", convert_to_tensor=True
            )

            m_score = (
                self._get_score(desc_emb, self.visual_pos, self.visual_neg)
                + self._get_score(desc_emb, self.novelty_pos, self.novelty_neg)
                + self._get_score(desc_emb, self.hype_pos, self.hype_neg)
            ) / 3.0

            margin = 0.0
            if "price" in p and "market_cost" in p and p["price"] > 0:
                margin = ((p["price"] - p["market_cost"]) / p["price"]) * 100.0

            trend_score = math.log1p(total_trend) * 2.5
            final = (m_score * 1.5) + (margin * 0.4) + trend_score

            processed.append(
                {
                    **p,
                    "_temp_trend": total_trend,
                    "_temp_margin": margin,
                    "_temp_final": final,
                }
            )

            print(f"{p['name'][:25]:<25} | {total_trend:<13} | {final:.2f}")

        top3_raw = sorted(processed, key=lambda x: x["_temp_final"], reverse=True)[:3]

        final_output: List[Dict[str, Any]] = []

        for item in top3_raw:
            rec_text = (
                f"Привлекательность: {item['_temp_final']:.1f}. "
                f"Спрос: {item['_temp_trend']} запросов. "
                f"Маржинальность: {int(item['_temp_margin'])}%."
            )

            clean_product = {k: v for k, v in item.items() if not k.startswith("_")}
            clean_product["recommendation"] = rec_text
            final_output.append(clean_product)

        output_file = "best_products.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)

        print("-" * 55)
        print(f"Сохранено {len(final_output)} топ-товаров в {output_file}")


if __name__ == "__main__":
    app = ProductAnalyzer("products.json")
    asyncio.run(app.run())
