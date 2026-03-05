"""
EntityEmbedding事前計算スクリプト

エンティティ名のベクトルを .npz ファイルに保存する。
SAFEパイプラインのSimEntインデックス構築を高速化（ワーカーはファイルから読み込むだけ）。

Usage:
  docker exec python-primekgqa-experiment python -m pipeline.safe.precompute_entity_embeddings --kg metaqa
  docker exec python-primekgqa-experiment python -m pipeline.safe.precompute_entity_embeddings --kg primekgqa
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from core.config import get_settings
from database.search import GraphPathFinder


BATCH_SIZE = 2000  # OpenAI Embeddings batch limit
CACHE_DIR = Path("/app/result/entity_embeddings")


def get_cache_path(kg_type: str) -> Path:
    return CACHE_DIR / f"{kg_type}.npz"


def precompute(kg_type: str):
    if not os.getenv("OPENAI_API_KEY"):
        settings = get_settings()
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    finder = GraphPathFinder(kg_type=kg_type)

    # 全エンティティ名を取得
    entity_names = finder.get_all_entity_names()
    print(f"Total entities: {len(entity_names)}", flush=True)

    cache_path = get_cache_path(kg_type)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # 既存キャッシュがあれば差分計算
    done_map: dict[str, np.ndarray] = {}
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        cached_names = data["names"].tolist()
        cached_vecs = data["vecs"]
        for i, name in enumerate(cached_names):
            done_map[name] = cached_vecs[i]
        print(f"Loaded existing cache: {len(done_map)} entities", flush=True)

    todo = [n for n in entity_names if n not in done_map]
    print(f"Already done: {len(done_map)}, remaining: {len(todo)}", flush=True)

    if todo:
        # バッチ処理
        total_batches = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_idx in range(total_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(todo))
            batch_names = todo[start:end]

            t0 = time.time()
            vectors = embeddings.embed_documents(batch_names)
            elapsed = time.time() - t0

            for name, vec in zip(batch_names, vectors):
                done_map[name] = np.array(vec, dtype=np.float32)

            print(
                f"  Batch {batch_idx + 1}/{total_batches}: "
                f"{len(batch_names)} entities, {elapsed:.1f}s",
                flush=True,
            )

    # .npz保存
    all_names = list(done_map.keys())
    all_vecs = np.stack([done_map[n] for n in all_names])
    np.savez(cache_path, names=np.array(all_names), vecs=all_vecs)
    print(f"\nSaved: {cache_path} ({len(all_names)} entities, {all_vecs.nbytes / 1e6:.1f}MB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Precompute entity embeddings")
    parser.add_argument(
        "--kg",
        type=str,
        required=True,
        choices=["metaqa", "primekgqa", "pcqa"],
        help="KG type",
    )
    args = parser.parse_args()
    precompute(args.kg)


if __name__ == "__main__":
    main()
