# src/config.py
from __future__ import annotations
import os, json, yaml, datetime
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

def _now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def _expand_vars(d: Dict[str, Any]) -> Dict[str, Any]:
    import re, copy
    pattern = re.compile(r"\$\{([^}]+)\}")

    def _get_by_path(root: Dict[str, Any], path: str):
        cur = root
        for k in path.split("."):
            cur = cur[k]
        return cur

    def _expand_once(obj, root):
        # 字典/列表遞迴
        if isinstance(obj, dict):
            return {k: _expand_once(v, root) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand_once(v, root) for v in obj]

        # 字串：支援 ${now} 與 ${a.b}
        if isinstance(obj, str):
            # 若是單一引用，直接回傳原型別（非字串）
            m = pattern.fullmatch(obj.strip())
            if m:
                key = m.group(1)
                if key == "now":
                    return _now_tag()
                try:
                    return _get_by_path(root, key)
                except Exception:
                    return obj  # 找不到就原樣保留

            # 其他情況（夾雜文字）逐一替換為字串
            def _repl(mm):
                key = mm.group(1)
                if key == "now":
                    return str(_now_tag())
                try:
                    val = _get_by_path(root, key)
                    return str(val)
                except Exception:
                    return mm.group(0)
            return pattern.sub(_repl, obj)
        return obj

    # 多輪展開直到收斂（最多 5 輪避免無限循環）
    cur = copy.deepcopy(d)
    for _ in range(5):
        nxt = _expand_once(cur, cur)
        if nxt == cur:
            break
        cur = nxt
    return cur


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
