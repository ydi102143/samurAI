# make_datasets.py
from __future__ import annotations
from pathlib import Path
import os, json, math, argparse, datetime, time, tempfile, sys
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# =========================================================
# 地域と都道府県（レベル=解放順）
# =========================================================
REGIONS: Dict[str, List[str]] = {
    "okinawa":  ["okinawa"],
    "kyushu":   ["fukuoka","saga","nagasaki","kumamoto","oita","miyazaki","kagoshima"],
    "shikoku":  ["tokushima","kagawa","ehime","kochi"],
    "chugoku":  ["tottori","shimane","okayama","hiroshima","yamaguchi"],
    "kinki":    ["mie","shiga","kyoto","osaka","hyogo","nara","wakayama"],
    "chubu":    ["niigata","toyama","ishikawa","fukui","yamanashi","nagano","gifu","shizuoka","aichi"],
    "kanto":    ["ibaraki","tochigi","gunma","saitama","chiba","tokyo","kanagawa"],
    "tohoku":   ["aomori","iwate","miyagi","akita","yamagata","fukushima"],
    "hokkaido": ["hokkaido"],
}
ALL_PROBLEMS = [p for ps in REGIONS.values() for p in ps]

def region_of(problem: str) -> str:
    for r, ps in REGIONS.items():
        if problem in ps: return r
    raise KeyError(problem)

REGION_LEVEL = {
    "okinawa": 1, "kyushu": 2, "shikoku": 3, "chugoku": 4,
    "kinki": 5, "chubu": 6, "kanto": 7, "tohoku": 8, "hokkaido": 9
}
PROBLEM_LEVEL = {p: REGION_LEVEL[r] for r, ps in REGIONS.items() for p in ps}

# =========================================================
# 県タグ / 県ごとの“優先したい特徴”ヒント
# （あなたの提示スクリプトを踏襲）
# =========================================================
TAGS = {
    "hokkaido":["ski","snow","tourism","alpine"],
    "aomori":["snow","port","agri"],
    "iwate":["snow","alpine","tourism"],
    "miyagi":["coastal","urban","port","tourism"],
    "akita":["snow","agri","coastal"],
    "yamagata":["snow","onsen","agri"],
    "fukushima":["inland","agri","alpine"],
    "ibaraki":["coastal","industrial"],
    "tochigi":["shrine","alpine","tourism"],
    "gunma":["onsen","alpine"],
    "saitama":["commuter","suburb"],
    "chiba":["airport","coastal","port"],
    "tokyo":["megacity","commuter","urban"],
    "kanagawa":["coastal","urban","port","tourism"],
    "niigata":["snow","port","rice","coastal"],
    "toyama":["snow","alpine"],
    "ishikawa":["tourism","coastal"],
    "fukui":["coastal","industrial"],
    "yamanashi":["fuji","alpine","tourism"],
    "nagano":["ski","alpine","snow"],
    "gifu":["alpine","tourism"],
    "shizuoka":["coastal","fuji","tourism","port"],
    "aichi":["industrial","port","urban"],
    "mie":["shrine","coastal","tourism"],
    "shiga":["lake","tourism"],
    "kyoto":["temple","tourism"],
    "osaka":["megacity","commuter","port"],
    "hyogo":["port","coastal","urban","tourism"],
    "nara":["temple","tourism"],
    "wakayama":["coastal","pilgrimage","tourism"],
    "tottori":["sand_dune","coastal","tourism"],
    "shimane":["shrine","coastal","tourism"],
    "okayama":["sunny","industrial","coastal"],
    "hiroshima":["tourism","port","coastal"],
    "yamaguchi":["strait","coastal"],
    "tokushima":["festival","coastal","mountain"],
    "kagawa":["udon","coastal","tourism"],
    "ehime":["islands","coastal","tourism"],
    "kochi":["surf","coastal","mountain"],
    "fukuoka":["urban","port","commuter"],
    "saga":["agri"],
    "nagasaki":["islands","port","coastal","tourism"],
    "kumamoto":["volcanic","agri"],
    "oita":["onsen","tourism","coastal"],
    "miyazaki":["beach","coastal","tourism"],
    "kagoshima":["volcanic","coastal","islands"],
    "okinawa":["beach","islands","typhoon","tourism"],
}
PROBLEM_HINTS = {
    "hokkaido": ["snowfall_cm","wind_speed_mps","snow_quality_index","ski_resort_visitors","temperature_c"],
    "aomori":   ["snowfall_cm","rainfall_mm","wind_speed_mps","ferry_cancel_count"],
    "iwate":    ["snowfall_cm","wind_speed_mps","rail_ridership","snow_quality_index"],
    "miyagi":   ["rail_ridership","mobility_inflow_index","tourist_inflow_count","event_local_flag","holiday"],
    "akita":    ["rainfall_mm","lodging_occupancy_ratio","ferry_cancel_count","typhoon_alert_flag"],
    "yamagata": ["temperature_c","rainfall_mm","snow_quality_index","onsen_visitors_count"],
    "fukushima":["mobility_inflow_index","temperature_c","wind_speed_mps","rail_ridership"],
    "tokyo":    ["mobility_inflow_index","rail_ridership","holiday","event_local_flag","rainfall_mm","temperature_c"],
    "kanagawa": ["beach_crowd_index","temperature_c","rail_ridership","holiday"],
    "chiba":    ["wind_speed_mps","flight_delay_count","rainfall_mm","typhoon_alert_flag"],
    "saitama":  ["rail_ridership","mobility_inflow_index","holiday","rainfall_mm"],
    "ibaraki":  ["event_local_flag","mobility_inflow_index","rainfall_mm","holiday"],
    "tochigi":  ["holiday","temperature_c","tourist_inflow_count","rail_ridership"],
    "gunma":    ["snowfall_cm","holiday","temperature_c","rail_ridership"],
    "niigata":  ["snowfall_cm","wind_speed_mps","temperature_c"],
    "toyama":   ["snowfall_cm","snow_quality_index","wind_speed_mps","rail_ridership"],
    "ishikawa": ["holiday","event_local_flag","rainfall_mm","lodging_occupancy_ratio","tourist_inflow_count"],
    "fukui":    ["ferry_cancel_count","rail_ridership","wind_speed_mps","rainfall_mm"],
    "yamanashi":["temperature_c","wind_speed_mps","holiday","special_local_index"],
    "nagano":   ["snowfall_cm","snow_quality_index","wind_speed_mps","temperature_c","ski_resort_visitors"],
    "gifu":     ["ferry_cancel_count","rainfall_mm","wind_speed_mps","event_local_flag"],
    "shizuoka": ["wind_speed_mps","flight_delay_count","rainfall_mm","typhoon_alert_flag"],
    "aichi":    ["rainfall_mm","ferry_cancel_count","event_local_flag","holiday"],
    "osaka":    ["holiday","temperature_c","rainfall_mm","rail_ridership","event_local_flag"],
    "hyogo":    ["temperature_c","holiday","rainfall_mm","tourist_inflow_count"],
    "kyoto":    ["holiday","event_local_flag","tourist_inflow_count","rainfall_mm","temple_shrine_visits"],
    "shiga":    ["rainfall_mm","ferry_cancel_count","lodging_occupancy_ratio"],
    "nara":     ["holiday","tourist_inflow_count","temperature_c","temple_shrine_visits"],
    "wakayama": ["event_local_flag","temperature_c","rainfall_mm","beach_crowd_index"],
    "tottori":  ["wind_speed_mps","temperature_c","holiday","special_local_index"],
    "shimane":  ["holiday","event_local_flag","rainfall_mm","rail_ridership"],
    "okayama":  ["temperature_c","rainfall_mm","lodging_occupancy_ratio","mobility_inflow_index"],
    "hiroshima":["ferry_cancel_count","holiday","rainfall_mm","tourist_inflow_count"],
    "yamaguchi":["wind_speed_mps","flight_delay_count","rainfall_mm","ferry_cancel_count"],
    "tokushima":["event_local_flag","holiday","rail_ridership"],
    "kagawa":   ["holiday","event_local_flag","rainfall_mm","tourist_inflow_count"],
    "ehime":    ["temperature_c","rainfall_mm","tourist_inflow_count","special_local_index"],
    "kochi":    ["rainfall_mm","typhoon_alert_flag","wind_speed_mps"],
    "fukuoka":  ["rail_ridership","event_local_flag","rainfall_mm","holiday"],
    "saga":     ["wind_speed_mps","temperature_c","holiday"],
    "nagasaki": ["holiday","rainfall_mm","event_local_flag"],
    "kumamoto": ["wind_speed_mps","flight_delay_count","special_local_index"],
    "oita":     ["holiday","rainfall_mm","temperature_c","onsen_visitors_count"],
    "miyazaki": ["beach_crowd_index","temperature_c","holiday"],
    "kagoshima":["wind_speed_mps","typhoon_alert_flag","special_local_index"],
    "okinawa":  ["beach_crowd_index","wind_speed_mps","typhoon_alert_flag","tourist_inflow_count"],
}

# =========================================================
# レベル→公開特徴量本数（L1=3 … L9=20）
# =========================================================
COLUMN_BUDGET = {1:3, 2:5, 3:7, 4:9, 5:12, 6:14, 7:16, 8:18, 9:20}

GLOBAL_FEATURE_POOL = [
    "holiday",
    "temperature_c","rainfall_mm","wind_speed_mps","snowfall_cm",
    "event_local_flag","typhoon_alert_flag",
    "mobility_inflow_index","tourist_inflow_count","lodging_occupancy_ratio",
    "rail_ridership","temple_shrine_visits","onsen_visitors_count",
    "beach_crowd_index","snow_quality_index","ski_resort_visitors",
    "ferry_cancel_count","flight_delay_count","special_local_index",
]

def _uniq_keep_order(seq):
    seen=set(); out=[]
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _select_features_for_level(problem: str, df_columns: List[str], lv: int) -> List[str]:
    budget = COLUMN_BUDGET.get(lv, 20)
    first = []
    if problem in PROBLEM_HINTS:
        for c in PROBLEM_HINTS[problem]:
            if c in df_columns:
                first.append(c)
    pool = [c for c in GLOBAL_FEATURE_POOL if c in df_columns]
    ordered = _uniq_keep_order(first + pool)
    return ordered[:budget]

# =========================================================
# 季節・地域パラメータ（あなたの提示を踏襲）
# =========================================================
CLIMATE_TEMP = {
    "okinawa": dict(amp=6.0,  bias=24.0, phase=+0.2),
    "kyushu":  dict(amp=10.0, bias=20.0, phase=+0.3),
    "shikoku": dict(amp=10.0, bias=19.0, phase=+0.3),
    "chugoku": dict(amp=11.0, bias=18.0, phase=+0.35),
    "kinki":   dict(amp=12.0, bias=17.0, phase=+0.35),
    "chubu":   dict(amp=13.0, bias=15.5, phase=+0.40),
    "kanto":   dict(amp=12.0, bias=16.5, phase=+0.40),
    "tohoku":  dict(amp=14.0, bias=12.0, phase=+0.45),
    "hokkaido":dict(amp=16.0, bias=9.0,  phase=+0.50),
}
TSUYU_MMDD = {
    "okinawa": ((5,10),(6,20)), "kyushu":((6,1),(7,10)), "shikoku":((6,1),(7,10)),
    "chugoku":((6,1),(7,10)), "kinki":((6,5),(7,15)), "chubu":((6,8),(7,18)),
    "kanto":((6,8),(7,20)), "tohoku":((6,20),(7,30)), "hokkaido":((7,1),(7,20)),
}
TYPHOON_PEAK = {"okinawa":1.0,"kyushu":0.7,"shikoku":0.6,"chugoku":0.4,"kinki":0.4,"chubu":0.3,"kanto":0.4,"tohoku":0.2,"hokkaido":0.15}

def _in_range(date: datetime.date, start: Tuple[int,int], end: Tuple[int,int]) -> bool:
    s = datetime.date(date.year, start[0], start[1])
    e = datetime.date(date.year, end[0], end[1])
    return s <= date <= e

# =========================================================
# ベース特徴量（季節/地域性を反映）
# =========================================================
def month_holiday_ratio(dates: np.ndarray):
    w = np.array([1 if d.weekday()>=5 else 0 for d in dates], dtype=float)
    m = np.array([d.month for d in dates])
    w += np.isin(m,[1,5,8,12]).astype(float)*0.5   # 連休・盆・年末年始ブースト
    return np.clip(w,0,1)

def make_base_features(problem: str, rows: int, rng: np.random.Generator) -> pd.DataFrame:
    region = region_of(problem)
    temp_cfg = CLIMATE_TEMP[region]
    today = datetime.date.today()
    dates = np.array([today - datetime.timedelta(days=rows - i - 1) for i in range(rows)])

    holiday = month_holiday_ratio(dates)
    months = np.array([d.month for d in dates])
    is_winter = np.isin(months, [12,1,2,3]).astype(float)

    ts_s, ts_e = TSUYU_MMDD[region]
    is_tsuyu = np.array([1.0 if _in_range(d, ts_s, ts_e) else 0.0 for d in dates])
    typh_base = np.isin(months, [8,9]).astype(float) * TYPHOON_PEAK[region]

    day_frac = np.array([d.timetuple().tm_yday/365.25 for d in dates])
    temp_core = temp_cfg["bias"] + temp_cfg["amp"] * np.sin(2*math.pi*(day_frac + temp_cfg["phase"]))
    temp_core += rng.normal(0, 0.8, rows)
    temperature_c = np.round(temp_core, 1)

    rain_level = 0.4*is_tsuyu + 0.6*typh_base + rng.beta(2,8,rows)*0.4
    if "snow" in TAGS.get(problem, []):
        rain_level -= 0.15*is_winter
    rainfall_mm = np.round(np.clip(rain_level, 0, 1)*45, 1)

    coast_bonus = 0.08 if "coastal" in TAGS.get(problem, []) else 0.0
    wind_speed_mps = np.round(np.clip(0.25 + 0.6*typh_base + coast_bonus + rng.normal(0.0,0.08,rows), 0, 1)*16, 1)

    coldness = np.clip((15 - (temperature_c - temp_cfg["bias"])) / 18, 0, 1)
    snow_factor = (np.isin(months,[12,1,2,3]).astype(float)) * coldness * (1.0 if region in ["tohoku","hokkaido","chubu"] or "snow" in TAGS.get(problem,[]) else 0.15)
    snowfall_cm = np.round(np.clip(snow_factor + rng.normal(0,0.05,rows), 0, 1)*35, 1)

    event_local_flag = (rng.random(rows) < 0.08).astype(int)
    typhoon_alert_flag = (rng.random(rows) < (0.10*typh_base + 0.01)).astype(int)

    mobility_inflow_index   = np.clip(0.55 + 0.25*(holiday - 0.5) + rng.normal(0, 0.08, rows), 0, 1)
    tourist_inflow_count    = np.clip(0.40*holiday + 0.25*event_local_flag + 0.20*np.clip(temperature_c-10,0,25)/25 - 0.15*rainfall_mm/45 + rng.normal(0,0.08,rows), 0, 1)
    lodging_occupancy_ratio = np.clip(0.50*holiday + 0.20*event_local_flag + 0.20*tourist_inflow_count + rng.normal(0,0.05,rows), 0, 1)
    rail_ridership          = np.clip(0.60*(1 - (holiday>0).astype(float)) - 0.08*rainfall_mm/45 + 0.15*mobility_inflow_index + rng.normal(0,0.05,rows), 0, 1)
    temple_shrine_visits    = np.clip(0.50*holiday + 0.20*event_local_flag + 0.10*np.clip(temperature_c-5,0,20)/20 + rng.normal(0,0.05,rows), 0, 1)
    onsen_visitors_count    = np.clip(0.35*holiday + 0.15*(1 - np.clip(temperature_c-0,0,20)/20) + rng.normal(0,0.05,rows), 0, 1)
    beach_crowd_index       = np.clip(0.60*np.clip(temperature_c-15,0,20)/20 - 0.25*wind_speed_mps/16 - 0.25*rainfall_mm/45 + 0.20*holiday + rng.normal(0,0.07,rows), 0, 1)
    snow_quality_index      = np.clip(0.65*(1 - np.clip(temperature_c-0,0,25)/25) + 0.35*snowfall_cm/35 - 0.10*rainfall_mm/45 + rng.normal(0,0.05,rows), 0, 1)
    ski_resort_visitors     = np.clip(0.55*snow_quality_index + 0.20*holiday - 0.10*rainfall_mm/45 + rng.normal(0,0.05,rows), 0, 1)
    ferry_cancel_count      = np.clip(0.50*wind_speed_mps/16 + 0.35*typhoon_alert_flag + 0.15*rainfall_mm/45 + rng.normal(0,0.03,rows), 0, 1)
    flight_delay_count      = np.clip(0.35*wind_speed_mps/16 + 0.25*rainfall_mm/45 + 0.30*typhoon_alert_flag + rng.normal(0,0.03,rows), 0, 1)

    special = np.zeros(rows)
    if "fuji" in TAGS.get(problem, []):        special += np.clip(0.4*np.clip(temperature_c-10,0,20)/20 - 0.2*rainfall_mm/45 + 0.2*holiday + rng.normal(0,0.05,rows), 0, 1)
    if "udon" in TAGS.get(problem, []):        special += np.clip(0.5*holiday + rng.normal(0,0.05,rows), 0, 1)
    if "sand_dune" in TAGS.get(problem, []):   special += np.clip(0.4*np.clip(temperature_c-10,0,20)/20 - 0.2*rainfall_mm/45 + rng.normal(0,0.05,rows), 0, 1)
    if "volcanic" in TAGS.get(problem, []):    special += np.clip(0.4*typhoon_alert_flag + 0.2*wind_speed_mps/16 + rng.normal(0,0.05,rows), 0, 1)

    df = pd.DataFrame({
        "date": dates,
        "holiday": np.where(holiday >= 0.5, "holiday", "weekday"),
        "temperature_c": np.round(temperature_c,1),
        "rainfall_mm": rainfall_mm,
        "wind_speed_mps": wind_speed_mps,
        "snowfall_cm": snowfall_cm,
        "event_local_flag": event_local_flag.astype(int),
        "typhoon_alert_flag": typhoon_alert_flag.astype(int),
        "mobility_inflow_index": np.round(mobility_inflow_index,3),
        "tourist_inflow_count": np.round(tourist_inflow_count,3),
        "lodging_occupancy_ratio": np.round(lodging_occupancy_ratio,3),
        "rail_ridership": np.round(rail_ridership,3),
        "temple_shrine_visits": np.round(temple_shrine_visits,3),
        "onsen_visitors_count": np.round(onsen_visitors_count,3),
        "beach_crowd_index": np.round(beach_crowd_index,3),
        "snow_quality_index": np.round(snow_quality_index,3),
        "ski_resort_visitors": np.round(ski_resort_visitors,3),
        "ferry_cancel_count": np.round(ferry_cancel_count,3),
        "flight_delay_count": np.round(flight_delay_count,3),
        "special_local_index": np.round(special,3),
    })
    return df

# =========================================================
# 目的変数 y（県タイプ別に回帰/分類）
# =========================================================
URBAN_REG = {"tokyo","kanagawa","osaka","aichi","saitama","fukuoka","miyagi","kyoto","nara"}
SNOW_CLS  = {"hokkaido","niigata","toyama","nagano","iwate"}
BEACH_CLS = {"okinawa"}
BEACH_REG = {"kanagawa","miyazaki","wakayama","chiba","shizuoka"}
TOUR_REG  = {"ishikawa","ehime","oita","yamanashi","gifu","tochigi","mie","shimane","tottori","kagawa"}

def _make_target(problem: str, df: pd.DataFrame, rng: np.random.Generator) -> Tuple[pd.Series, str]:
    def nz(col, default=0.0):
        s = df.get(col)
        if s is None: return pd.Series([default]*len(df))
        return pd.to_numeric(s, errors="coerce").fillna(default)

    temp_n  = (nz("temperature_c") - 5.0) / 25.0
    rain_n  = nz("rainfall_mm") / 45.0
    wind_n  = nz("wind_speed_mps") / 16.0
    snow_n  = nz("snowfall_cm") / 35.0
    mob_n   = nz("mobility_inflow_index")
    rail_n  = nz("rail_ridership")
    tour_n  = nz("tourist_inflow_count")
    lodge_n = nz("lodging_occupancy_ratio")
    beach_n = nz("beach_crowd_index")
    snowq_n = nz("snow_quality_index")
    fdel_n  = nz("flight_delay_count")
    ferry_n = nz("ferry_cancel_count")
    typh    = nz("typhoon_alert_flag")
    event_f = nz("event_local_flag")
    hol     = (df.get("holiday") == "holiday").astype(float).fillna(0.0)

    # 都市圏: 交通・流入の回帰
    if problem in URBAN_REG:
        y = 0.45*rail_n + 0.25*mob_n + 0.15*hol + 0.05*event_f - 0.10*rain_n
        y = np.clip(y + rng.normal(0, 0.03, len(df)), 0, 1)
        return pd.Series(y).round(3), "regression"

    # 豪雪地: リスク分類
    if problem in SNOW_CLS:
        cold = np.clip(1.0 - temp_n, 0, 1)
        risk = 0.45*snow_n + 0.25*wind_n + 0.15*cold + 0.05*typh - 0.05*rain_n
        risk = np.clip(risk + rng.normal(0, 0.05, len(df)), 0, 1)
        thr  = float(np.quantile(risk, 0.6))  # 正例~40%
        y = (risk >= thr).astype(int)
        return pd.Series(y), "classification"

    # 沖縄: 台風/風の分類
    if problem in BEACH_CLS:
        risk = 0.45*wind_n + 0.25*typh + 0.20*rain_n - 0.10*temp_n
        risk = np.clip(risk + rng.normal(0, 0.05, len(df)), 0, 1)
        thr  = float(np.quantile(risk, 0.5))
        y = (risk >= thr).astype(int)
        return pd.Series(y), "classification"

    # ビーチ観光回帰
    if problem in BEACH_REG:
        y = 0.40*np.clip(temp_n, 0, 1) - 0.20*wind_n - 0.15*rain_n + 0.20*hol + 0.10*rail_n
        y = np.clip(y + rng.normal(0, 0.04, len(df)), 0, 1)
        return pd.Series(y).round(3), "regression"

    # 観光地回帰
    if problem in TOUR_REG:
        y = 0.35*hol + 0.20*event_f + 0.20*tour_n + 0.10*lodge_n + 0.05*np.clip(temp_n, 0, 1) - 0.10*rain_n
        y = np.clip(y + rng.normal(0, 0.03, len(df)), 0, 1)
        return pd.Series(y).round(3), "regression"

    # デフォルト: 交通の乱れ分類
    risk = 0.35*fdel_n + 0.25*ferry_n + 0.20*wind_n + 0.10*rain_n + 0.10*typh
    risk = np.clip(risk + rng.normal(0, 0.05, len(df)), 0, 1)
    thr  = float(np.quantile(risk, 0.6))
    y = (risk >= thr).astype(int)
    return pd.Series(y), "classification"

# =========================================================
# 欠損注入（学習課題のため）
# =========================================================
MISSING_PLAN = {
    "rainfall_mm": 0.04, "event_local_flag": 0.03, "mobility_inflow_index": 0.04,
    "tourist_inflow_count": 0.05, "lodging_occupancy_ratio": 0.05, "rail_ridership": 0.05,
    "temple_shrine_visits": 0.04, "onsen_visitors_count": 0.04, "beach_crowd_index": 0.05,
    "snow_quality_index": 0.05, "ski_resort_visitors": 0.05, "ferry_cancel_count": 0.05,
    "flight_delay_count": 0.05, "special_local_index": 0.05,
}

def inject_missing(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    for col, rate in MISSING_PLAN.items():
        if col not in df.columns: continue
        k = max(1, int(n * rate))
        idx = rng.choice(n, size=k, replace=False)
        if pd.api.types.is_numeric_dtype(df[col]):
            df.iloc[idx, df.columns.get_loc(col)] = np.nan
        else:
            df.iloc[idx, df.columns.get_loc(col)] = ""
    if "holiday" in df.columns:
        k = max(1, int(n * 0.01))
        idx = rng.choice(n, size=k, replace=False)
        df.iloc[idx, df.columns.get_loc("holiday")] = ""
    return df

# =========================================================
# 1県ぶん生成（レベルに応じて公開列を間引き）
# =========================================================
def build_one(problem: str, rows=730, seed=42) -> Tuple[pd.DataFrame, int, str]:
    rng = np.random.default_rng(abs(hash((problem, seed))) % (2**32))
    df = make_base_features(problem, rows, rng)
    y, task_type = _make_target(problem, df, rng)
    df["y"] = y
    df = inject_missing(df, rng)

    lv = PROBLEM_LEVEL.get(problem, 1)
    feature_cols = [c for c in df.columns if c not in ("date","y")]
    selected_feats = _select_features_for_level(problem, feature_cols, lv)
    cols = ["date"] + selected_feats + ["y"]
    return df[cols], lv, task_type

# =========================================================
# Lv解放：機能 & モデル（あなたのLv1〜Lv9仕様に完全準拠）
# =========================================================
def allowed_for_level(level: int, task: str):
    """
    Lv1：最短「前処理→学習→評価→提出」
    Lv2：EDAプレビュー / schema
    Lv3：相関・ヒスト・散布図 + One-Hot
    Lv4：スケーリング + CV
    Lv5：特徴量重要度 + ランダムサーチ + ベスト保存
    Lv6：GBM(Ada/Gradient)入門
    Lv7：XGBoost
    Lv8：ROC/PR + Youden’s J しきい値提案
    Lv9：スタッキング + モデルカードCSV
    """
    feats = ["Preprocess","Train","Evaluate","Submit"]  # Lv1
    if level >= 2: feats += ["EDAPreview","Schema"]
    if level >= 3: feats += ["Correlation","Histogram","Scatter","OneHot"]
    if level >= 4: feats += ["Scaling","CrossValidation"]
    if level >= 5: feats += ["FeatureImportance","RandomSearch","BestModelSave"]
    if level >= 6: feats += ["GBM_Ada","GBM_Gradient"]
    if level >= 7: feats += ["XGBoost"]
    if level >= 8: feats += ["ROC_PR_Curves","YoudenJ_Thresholding"]
    if level >= 9: feats += ["Stacking","ModelCardCSV"]
    allowed_features = sorted(set(feats + ["EDA","Predict"]))  # 互換キー

    if task == "classification":
        models = ["LogisticRegression"]                       # Lv1
        if level >= 3: models += ["DecisionTreeClassifier","RandomForestClassifier","KNeighborsClassifier"]
        if level >= 6: models += ["AdaBoostClassifier","GradientBoostingClassifier"]
        if level >= 7: models += ["XGBClassifier"]
        if level >= 9: models += ["StackingClassifier"]
    else:
        models = ["LinearRegression"]                         # Lv1
        if level >= 3: models += ["DecisionTreeRegressor","RandomForestRegressor","KNeighborsRegressor"]
        if level >= 6: models += ["AdaBoostRegressor","GradientBoostingRegressor"]
        if level >= 7: models += ["XGBRegressor"]
        if level >= 9: models += ["StackingRegressor"]
    return allowed_features, sorted(set(models))

# =========================================================
# メタ出力（problem_meta.json & pref_meta.json）
# =========================================================
def target_meta_for(problem: str) -> Tuple[dict, str]:
    # build_oneのtask_typeと整合（安全のため再計算ルールを共有）
    if problem in URBAN_REG or problem in BEACH_REG or problem in TOUR_REG:
        typ, metric = "regression", "r2"
    elif problem in SNOW_CLS or problem in BEACH_CLS:
        typ, metric = "classification", "accuracy"
    else:
        typ, metric = "classification", "accuracy"
    return {"name":"y","type":typ}, metric

# =========================================================
# 安全CSV書き込み（OneDrive/Excelロック対策）
# =========================================================
def safe_write_csv(df: pd.DataFrame, out_csv: Path, max_retries: int = 8, retry_wait: float = 0.6):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=out_csv.stem + "_", suffix=".tmp", dir=str(out_csv.parent))
    os.close(fd)
    tmp_path = Path(tmp_path)
    df2 = df.copy()
    if "date" in df2.columns:
        df2["date"] = pd.to_datetime(df2["date"]).dt.strftime("%Y-%m-%d")
    last_err = None
    for _ in range(max_retries):
        try:
            df2.to_csv(tmp_path, index=False, encoding="utf-8")
            os.replace(tmp_path, out_csv)
            return
        except Exception as e:
            last_err = e
            time.sleep(retry_wait)
    try:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    finally:
        pass
    raise PermissionError(f"Failed to write {out_csv}: {last_err}")

# =========================================================
# CLI
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser(description="samurAI datasets generator (seasonality + levels)")
    ap.add_argument("--root", type=str, default="storage", help="出力ルート（既定: storage）")
    ap.add_argument("--rows", type=int, default=730, help="各県の行数（既定: 730 = 2年弱の時系列）")
    ap.add_argument("--seed", type=int, default=42, help="乱数シード")
    ap.add_argument("--only", nargs="*", default=None, help="生成対象の県スラッグ（空白区切り）例: tokyo osaka okinawa")
    ap.add_argument("--all", action="store_true", help="全県生成")
    ap.add_argument("--overwrite", action="store_true", help="既存CSVがあっても上書き")
    return ap.parse_args()

def main():
    args = parse_args()
    out_root = Path(args.root)

    problems = ALL_PROBLEMS if (args.all or not args.only) else [p for p in ALL_PROBLEMS if p in set(args.only)]
    if not problems:
        print("生成対象がありません。--all か --only を指定してください。", file=sys.stderr)
        sys.exit(1)

    ds_root = out_root / "datasets"
    ds_root.mkdir(parents=True, exist_ok=True)
    # 県ごとCSV
    meta_pref: Dict[str, dict] = {}
    for problem in problems:
        df, lv, task_type = build_one(problem, rows=args.rows, seed=args.seed)
        region = region_of(problem)
        out_dir = ds_root / region
        out_csv = out_dir / f"{problem}.csv"
        latest_csv = out_dir / "_processed" / problem / "latest.csv"

        if (not args.overwrite) and out_csv.exists():
            print(f"[SKIP] exists: {out_csv}  (--overwriteで上書き)", file=sys.stderr)
        else:
            safe_write_csv(df, out_csv)
            safe_write_csv(df, latest_csv)
            print(f"[OK] {out_csv} rows={len(df)} cols={len(df.columns)} lv={lv} (latest->{latest_csv})")

        # メタ（県単位）
        target, metric = target_meta_for(problem)
        feats, models = allowed_for_level(lv, task_type)
        meta_pref[problem] = {
            "region": region,
            "level": lv,
            "target": target,                       # {"name":"y","type":...}
            "metric": metric,                       # accuracy / r2
            "greater_is_better": metric not in ["logloss","rmse","mae"],
            "allowed_features": feats,
            "allowed_models": models,
            "rows": int(len(df)),
            "columns": [c for c in df.columns if c!="date"],
        }

    # メタ一括出力
    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    pref_meta_path = meta_dir / "pref_meta.json"
    pref_meta_path.write_text(json.dumps(meta_pref, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"meta -> {pref_meta_path}")

    # README（軽い説明）
    readme = f"""# samurAI datasets

- 生成日時: {datetime.datetime.now().isoformat(timespec='seconds')}
- 保存先: storage/datasets/<region>/<pref>.csv（県別） / _processed/<pref>/latest.csv（最新の複製）
- メタ: storage/meta/pref_meta.json（レベル/Lv解放・モデル・行数・列一覧）

## 列の意味（抜粋）
- date: 日付（連続日）
- holiday: "holiday"/"weekday"
- weather系: temperature_c, rainfall_mm, wind_speed_mps, snowfall_cm
- 観光/移動系: mobility_inflow_index, tourist_inflow_count, lodging_occupancy_ratio, rail_ridership, temple_shrine_visits, onsen_visitors_count, beach_crowd_index, snow_quality_index, ski_resort_visitors
- イベント/警報系: event_local_flag, typhoon_alert_flag, ferry_cancel_count, flight_delay_count, special_local_index
- y: ターゲット（県タイプにより regression(0-1) または classification(0/1)）
"""
    (ds_root / "README.md").write_text(readme, encoding="utf-8")

if __name__ == "__main__":
    main()
