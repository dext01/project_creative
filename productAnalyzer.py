# productAnalyzer.py
import json
from typing import List, Dict, Any

import pandas as pd

from main import select_top_products, _safe_float  # _safe_float уже в main


def load_catalog(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "products" in data:
            return data["products"]
        return data
    else:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")


def save_best_products(input_path: str, output_path: str = "best_products.json") -> None:
    catalog = load_catalog(input_path)
    top_products = select_top_products(catalog, k=3)
    result = []
    for p in top_products:
        result.append(
            {
                "name": p.name,
                "category": p.category,
                "price": p.price,
                "margin": p.margin,
                "tags": p.tags,
                "description": p.description,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Сохранено {len(result)} товаров в {output_path}")


if __name__ == "__main__":
    # пример: python productAnalyzer.py
    save_best_products("products.json", "best_products.json")
