# src/config.py
from __future__ import annotations
import os, json, yaml, datetime
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

def _now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def _expand_vars(d: Dict[str, Any]) -> Dict[str, Any]:
    """支援 ${now} 與 ${a.b} 這種引用。"""
    import re
    pattern = re.compile(r"\$\{([^}]+)\}")
    def resolve(expr: str, root: Dict[str, Any]) -> str:
        if expr == "now": 
            return _now_tag()
        # 支援 ${data.root} 形式
        cur = root
        for k in expr.split("."):
            cur = cur[k]
        return str(cur)

    def recur(x):
        if isinstance(x, dict):
            return {k: recur(v) for k, v in x.items()}
        if isinstance(x, list):
            return [recur(v) for v in x]
        if isinstance(x, str):
            def repl(m): return resolve(m.group(1), d)
            return pattern.sub(repl, x)
        return x
    return recur(d)

def _deep_update(base: Dict[str, Any], override: Dict[str, Any] | None) -> Dict[str, Any]:
    if not override: 
        return base
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base

def _parse_overrides(pairs):
    """將 ['train.epochs=10','data.root=data'] 轉成 dict"""
    out: Dict[str, Any] = {}
    for pair in pairs or []:
        key, val = pair.split("=", 1)
        # 自動轉型
        if val.lower() in ("true","false"):
            val = val.lower() == "true"
        else:
            try: val = int(val)
            except:
                try: val = float(val)
                except: pass
        cur = out
        ks = key.split(".")
        for k in ks[:-1]:
            cur = cur.setdefault(k, {})
        cur[ks[-1]] = val
    return out

def load_cfg(cfg_path: str, overrides: list[str] | None = None) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # optional env override by HOST_ALIAS
    host = os.environ.get("HOST_ALIAS", "")
    env_map = {
        "3080": "configs/env/3080.yaml",
        "1650": "configs/env/1650.yaml",
    }
    if host in env_map and Path(env_map[host]).exists():
        with open(env_map[host], "r", encoding="utf-8") as f:
            _deep_update(cfg, yaml.safe_load(f))

    _deep_update(cfg, _parse_overrides(overrides))
    cfg = _expand_vars(cfg)
    # 確保輸出目錄存在
    outdir = Path(cfg["project"]["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    # 保存最終設定（可重現）
    with open(outdir / "config.final.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return cfg
