from __future__ import annotations
from typing import List, Dict
import pandas as pd
import numpy as np
from io import BytesIO
from fastapi.responses import Response
from matplotlib import pyplot as plt

def summarize(df: pd.DataFrame) -> list[dict]:
    out=[]
    for c in df.columns:
        s=df[c]
        dtype = "number" if pd.api.types.is_numeric_dtype(s) else ("category" if s.dtype=="object" else str(s.dtype))
        out.append({"name": c, "dtype": dtype, "na_count": int(s.isna().sum()), "unique": int(s.nunique(dropna=True))})
    return out

def resp_png(fig) -> Response:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig); buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")

def plot_missing(df: pd.DataFrame, title: str) -> Response:
    miss = df.isna().mean()
    fig = plt.figure(figsize=(max(5, len(miss)*0.5), 3))
    ax = fig.add_subplot(111)
    ax.bar(range(len(miss)), miss.values)
    ax.set_xticks(range(len(miss)))
    ax.set_xticklabels(miss.index, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("missing rate"); ax.set_title(title)
    fig.tight_layout()
    return resp_png(fig)