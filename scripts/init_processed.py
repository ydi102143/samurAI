# scripts/init_processed.py
from __future__ import annotations
from pathlib import Path
import json, shutil, argparse, sys

ROOT = Path("storage/datasets")
META = ROOT / "_meta" / "pref_meta.json"

def load_targets():
    targets = []
    # 1) meta があれば優先（region/pref を正確に辿れる）
    if META.exists():
        try:
            m = json.loads(META.read_text(encoding="utf-8"))
            for pref, v in m.items():
                region = (v or {}).get("region")
                if region:
                    targets.append((region, pref))
        except Exception as e:
            print(f"[WARN] meta parse failed: {e}", file=sys.stderr)

    # 2) フォルダ走査で補完（metaが空でも動く）
    for region_dir in ROOT.glob("*"):
        if not region_dir.is_dir() or region_dir.name == "_meta":
            continue
        for csv_path in region_dir.glob("*.csv"):
            pref = csv_path.stem
            key = (region_dir.name, pref)
            if key not in targets:
                targets.append(key)
    return targets

def ensure_processed(region: str, pref: str, force: bool, from_raw: bool):
    raw = ROOT / region / f"{pref}.csv"
    out_dir = ROOT / region / "_processed" / pref
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / "latest.csv"
    pipe   = out_dir / "pipeline.json"
    feat   = out_dir / "feature_selection.json"

    if not raw.exists():
        return False, f"raw not found: {raw}"

    # latest.csv: 既存を残したい場合は --force で上書き
    if (not latest.exists()) or force or from_raw:
        shutil.copyfile(raw, latest)

    # からの定義を用意（無ければ作る）
    if not pipe.exists():
        pipe.write_text("[]", encoding="utf-8")
    if not feat.exists():
        feat.write_text(json.dumps({"num_cols": [], "cat_cols": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    return True, str(latest)

def main():
    ap = argparse.ArgumentParser(description="Initialize _processed/latest.csv for all prefectures.")
    ap.add_argument("--force", action="store_true", help="既存 latest.csv があっても上書きコピーする")
    ap.add_argument("--from-raw", action="store_true", help="必ず raw から latest.csv を作り直す（= --force と同義）")
    args = ap.parse_args()

    targets = load_targets()
    if not targets:
        print("[INFO] no targets found under storage/datasets")
        return

    ok, ng = 0, 0
    for region, pref in sorted(targets):
        success, msg = ensure_processed(region, pref, force=args.force or args.from_raw, from_raw=args.from_raw)
        if success:
            ok += 1
            print(f"[OK] {region}/{pref} → {msg}")
        else:
            ng += 1
            print(f"[NG] {region}/{pref} → {msg}")

    print(f"\nDone. success={ok}, failed={ng}")

if __name__ == "__main__":
    main()