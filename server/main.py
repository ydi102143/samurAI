from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
import pandas as pd, io, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
app = FastAPI()
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "train.csv"
@app.get("/health")
def health():
    return {"status": "ok"}
def _load_df():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    import numpy as np
    return pd.DataFrame({"Age": np.random.normal(30,10,500),
                         "Fare": np.random.exponential(30,500),
                         "Survived": np.random.randint(0,2,500)})
@app.post("/eda/histogram")
def eda_histogram(col: str = Body(..., embed=True)):
    df = _load_df()
    if col not in df.columns:
        return JSONResponse({"error": f"{col} not in columns", "hint": list(df.columns)}, status_code=400)
    fig = plt.figure()
    df[col].dropna().hist(bins=30); plt.title(f"Histogram: {col}")
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return {"image_base64": base64.b64encode(buf.getvalue()).decode("utf-8")}
