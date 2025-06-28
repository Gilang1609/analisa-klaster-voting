"""
analyzer.py
-----------

Modul ini menerima daftar teks + embeddings,
mengelompokkan alasan menggunakan K‑Means, lalu
menghasilkan ringkasan naratif & kata‑kunci per klaster
dengan bantuan Cohere.

Fungsi utama:
    analisa(texts: list[str], embeddings: list[list[float]]) -> dict

Pastikan variabel lingkungan COHERE_KEY berisi API‑key Cohere Anda,
atau langsung ubah nilai defaultnya di bawah (tidak disarankan).
"""

import os
import json
import numpy as np
import requests
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

# ===========================================================
# ✨ Konfigurasi
# ===========================================================

COHERE_KEY = os.getenv("COHERE_KEY") or "VVxxzNRSMO2xMgk1HQjsaFu88jX05j457Qi9507u"

# ===========================================================
# ✨ Helper ‑ Cohere
# ===========================================================


def cohere_chat(
    prompt: str,
    model: str = "command-r-plus",
    max_tokens: int = 300,
    temperature: float = 0.7,
) -> str:
    """
    Memanggil endpoint /chat Cohere dan mengembalikan teks respons.
    """
    url = "https://api.cohere.ai/v1/chat"
    headers = {
        "Authorization": f"Bearer {COHERE_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "message": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    resp = requests.post(url, headers=headers, json=data, timeout=60)
    resp.raise_for_status()
    return resp.json().get("text", "").strip()


def cohere_keywords_from_narasi(narasi: str, jumlah: int = 5) -> list[str]:
    """
    Meminta Cohere meringkas narasi menjadi N kata kunci tematik.
    """
    pmt = (
        f"Berdasarkan narasi berikut, ringkaslah menjadi {jumlah} kata kunci "
        "tematik utama yang mewakili isi narasi secara bermakna.\n\n"
        f"Narasi:\n{narasi}\n\nKata kunci:"
    )
    try:
        result = cohere_chat(pmt)
        keywords = [
            w.strip(" ,.-") for w in result.splitlines() if w.strip()
        ]
        # Jika Cohere merespons satu baris dipisah koma
        if len(keywords) == 1 and "," in keywords[0]:
            keywords = [k.strip() for k in keywords[0].split(",")]
        return keywords[:jumlah]
    except Exception as e:
        return [f"(Gagal ambil kata kunci: {e})"]


# ===========================================================
# ✨ Stop‑words & TF‑IDF helper
# ===========================================================

_stopwords_id = set(
    """
    saya kamu dia mereka kami kita anda tidak ya yang dengan dalam untuk pada
    dari oleh ini itu adalah ke telah sudah akan masih bisa dapat karena agar
    bila jika sebagai tentang bahwa jadi pun lebih hanya semua sangat lalu namun
    """.split()
)


def top_keywords(texts: list[str], top_n: int = 5) -> list[str]:
    """
    Mengambil kata kunci TF‑IDF teratas dari kumpulan teks.
    """
    if not texts:
        return []
    vect = TfidfVectorizer(token_pattern=r"\b\w+\b")
    X = vect.fit_transform(texts)
    terms = vect.get_feature_names_out()
    mean_vect = X.mean(axis=0).A1
    scored = [
        (t, s)
        for t, s in zip(terms, mean_vect)
        if t.lower() not in _stopwords_id
    ]
    top = sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
    return [t for t, _ in top]


# ===========================================================
# ✨ Fungsi utama
# ===========================================================


def analisa(texts: list[str], embeddings: list[list[float]]) -> dict:
    """
    Menjalankan seluruh pipeline:
        1. Normalisasi embeddings
        2. Menentukan jumlah klaster optimal (silhouette score)
        3. K‑Means clustering
        4. Ringkasan naratif & kata kunci (Cohere + TF‑IDF)
        5. Mengembalikan struktur JSON seperti contoh di README
    """
    # Validasi input
    if not texts or not embeddings:
        return {"error": "texts atau embeddings kosong"}

    embeddings = np.array(embeddings, dtype="float32")

    # Kasus <2 alasan → langsung unik
    if len(texts) < 2:
        return {
            "clusters": [],
            "unik": {"jumlah": len(texts), "contoh": texts[:5]},
        }

    # Normalisasi vektor
    embeddings = normalize(embeddings)

    # Cari K optimal 2..10 (atau maksimal jumlah data)
    scores: list[tuple[int, float]] = []
    for k in range(2, min(10, len(texts)) + 1):
        if k >= len(embeddings):
            break
        mdl = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(
            embeddings
        )
        sc = silhouette_score(embeddings, mdl.labels_)
        scores.append((k, sc))

    best_k = max(scores, key=lambda t: t[1])[0] if scores else 1

    # Clustering final
    final_mdl = KMeans(n_clusters=best_k, random_state=42, n_init="auto").fit(
        embeddings
    )
    labels = final_mdl.labels_

    # Kelompokkan alasan
    clusters: dict[int, list[str]] = {}
    for lbl, txt in zip(labels, texts):
        clusters.setdefault(lbl, []).append(txt)

    # Kumpulan alasan tunggal → unik
    unik = [txt for grp in clusters.values() if len(grp) == 1 for txt in grp]

    # Urutkan cluster terpadat dulu
    sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))

    result_clusters = []
    for i, (lbl, alasan_list) in enumerate(sorted_clusters, 1):
        if len(alasan_list) <= 1:
            continue

        contoh = alasan_list[:5]
        prompt = (
            "Berikan ringkasan naratif yang jelas, padat, bermakna "
            "berdasarkan kumpulan alasan berikut:\n"
            + "\n".join(f"- {a}" for a in contoh)
            + "\nNarasi:"
        )

        try:
            narasi = cohere_chat(prompt)
            kata_kunci_narasi = cohere_keywords_from_narasi(narasi)
        except Exception as e:
            narasi = f"(Gagal generate narasi: {e})"
            kata_kunci_narasi = [f"(Gagal ambil kata kunci: {e})"]

        result_clusters.append(
            {
                "judul": f"🟢 Klaster #{i} ({len(alasan_list)} alasan)",
                "narasi": narasi,
                "kata_kunci_alasan": top_keywords(alasan_list),
                "kata_kunci_narasi": kata_kunci_narasi,
                "contoh": contoh,
            }
        )

    return {
        "clusters": result_clusters,
        "unik": {"jumlah": len(unik), "contoh": unik[:5]},
    }


# ===========================================================
# ✨ CLI fallback  (opsional)
# ===========================================================

if __name__ == "__main__":
    """
    Penggunaan:
        python analyzer.py input.json

    File JSON harus berisi:
    {
        "texts": ["...", "..."],
        "embeddings": [[0.1, 0.2, ...], [...]]
    }
    """
    if len(sys.argv) < 2:
        print("Usage: python analyzer.py data.json")
        exit(1)

    import sys

    path = sys.argv[1]
    with open(path, "r") as f:
        payload = json.load(f)

    out = analisa(payload.get("texts", []), payload.get("embeddings", []))
    print(json.dumps(out, ensure_ascii=False, indent=2))
