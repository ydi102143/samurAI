# samurAI datasets

- 生成日時: 2025-09-12T00:47:08
- 保存先: storage/datasets/<region>/<pref>.csv（県別） / _processed/<pref>/latest.csv（最新の複製）
- メタ: storage/meta/pref_meta.json（レベル/Lv解放・モデル・行数・列一覧）

## 列の意味（抜粋）
- date: 日付（連続日）
- holiday: "holiday"/"weekday"
- weather系: temperature_c, rainfall_mm, wind_speed_mps, snowfall_cm
- 観光/移動系: mobility_inflow_index, tourist_inflow_count, lodging_occupancy_ratio, rail_ridership, temple_shrine_visits, onsen_visitors_count, beach_crowd_index, snow_quality_index, ski_resort_visitors
- イベント/警報系: event_local_flag, typhoon_alert_flag, ferry_cancel_count, flight_delay_count, special_local_index
- y: ターゲット（県タイプにより regression(0-1) または classification(0/1)）
