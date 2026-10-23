"""网页「设置」：读取 / 保存配置、测试模型连接、测试检索来源。

保存在 data/settings.json，优先于 .env，保存后立即生效（不用重启）。
密钥只写不读：接口只返回是否已设置和末 4 位，不返回明文。
"""
from __future__ import annotations

import os

import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import config, llm, scholar
from ..security import settings_locked
from ..config import OVERRIDABLE_KEYS, SECRET_KEYS, TIERS, get_settings, load_overrides, save_overrides, value_source

router = APIRouter(prefix="/api/settings", tags=["settings"])

_INT_KEYS = {"SEARCH_CACHE_DAYS", "TRACE_RETENTION_DAYS", "TRACE_DETAIL_DAYS", *(f"MODEL_{t.upper()}_TIMEOUT" for t in TIERS)}
_SOURCES = {"semantic_scholar", "arxiv", "openalex"}


def _current(name: str) -> str | None:
    return config._env(name)  # 已合并 settings.json 与 .env


def _view() -> dict:
    values = {}
    for name in sorted(OVERRIDABLE_KEYS):
        v = _current(name)
        item: dict = {"source": value_source(name), "set": v is not None}
        if name in SECRET_KEYS:
            item["hint"] = f"····{v[-4:]}" if v and len(v) > 8 else ("已设置" if v else None)
        else:
            item["value"] = v
        values[name] = item
    # 旧版字段（只在 .env 里出现）也算作已配置的 key
    if not values["LLM_API_KEY"]["set"] and os.getenv("OPENAI_API_KEY"):
        k = os.getenv("OPENAI_API_KEY") or ""
        values["LLM_API_KEY"] = {"source": "env", "set": True, "hint": f"····{k[-4:]}（OPENAI_API_KEY）"}
    if not values["LLM_API_BASE"]["set"] and os.getenv("OPENAI_BASE_URL"):
        values["LLM_API_BASE"] = {"source": "env", "set": True, "value": os.getenv("OPENAI_BASE_URL")}
    s = get_settings()
    return {
        "values": values,
        "tiers": llm.tier_info(),
        "llm_ready": s.llm_ready,
        "fake": s.llm_fake,
        "settings_file": str(config.settings_path()),
        "locked": settings_locked(),
    }


@router.get("")
def read_settings():
    return _view()


class SettingsIn(BaseModel):
    values: dict[str, str | None]


@router.put("")
def write_settings(body: SettingsIn):
    if settings_locked():
        raise HTTPException(403, "管理员已锁定设置（ARIA_LOCK_SETTINGS），只能查看不能修改")
    changes: dict[str, str | None] = {}
    for k, v in body.values.items():
        if k not in OVERRIDABLE_KEYS:
            raise HTTPException(400, f"不支持的设置项：{k}")
        if v is not None and str(v).strip() and str(v).strip() != config.EXPLICIT_EMPTY:
            v = str(v).strip()
            if k in _INT_KEYS and not v.isdigit():
                raise HTTPException(400, f"{k} 需要是整数")
            if k in {"TRACE_RETENTION_DAYS", "TRACE_DETAIL_DAYS"} and not 1 <= int(v) <= 365:
                raise HTTPException(400, "追踪保留天数需要在 1 到 365 之间")
            if k == "SEARCH_SOURCES":
                srcs = [x.strip() for x in v.split(",") if x.strip()]
                if not srcs or any(x not in _SOURCES for x in srcs):
                    raise HTTPException(400, "检索来源只能是 semantic_scholar / arxiv / openalex")
                v = ",".join(srcs)
            if k == "MODEL_PRICES":
                try:
                    data = json.loads(v)
                    assert isinstance(data, dict)
                    for m, pr in data.items():
                        for f in ("input", "output"):
                            assert float(pr[f]) >= 0
                        if "cached" in pr:
                            assert float(pr["cached"]) >= 0
                except Exception:
                    raise HTTPException(400, '自定义价格格式不对：应为 {"模型名": {"input": 2, "cached": 0.2, "output": 8}}，单位 元/百万 tokens') from None
                v = json.dumps(data, ensure_ascii=False)
            if k == "PDF_PARSER" and v not in ("pymupdf", "grobid", "mineru"):
                raise HTTPException(400, "PDF 解析器只能是 pymupdf / grobid / mineru")
            if k in ("GROBID_URL", "MINERU_URL") and not v.startswith(("http://", "https://")):
                raise HTTPException(400, f"{k} 需要以 http:// 或 https:// 开头")
            if k in {"LLM_FAKE", "TRACE_CAPTURE"}:
                v = "1" if v.lower() in {"1", "true", "yes", "on"} else "0"
        changes[k] = v
    save_overrides(changes)
    return _view()


class TestLLMIn(BaseModel):
    tier: str | None = None  # 测试某一档（使用已保存的配置，可被下面的字段临时覆盖）
    model: str | None = None
    api_base: str | None = None
    api_key: str | None = None  # 不传则用已保存的 key


@router.post("/test-llm")
def test_llm(body: TestLLMIn):
    if settings_locked() and (body.model or body.api_base is not None or body.api_key):
        # 锁定时只能测试已保存的配置：否则可以把服务器上的密钥发到任意地址
        raise HTTPException(403, "管理员已锁定设置，只能测试已保存的模型配置")
    s = get_settings()
    base_tier = s.tiers.get(body.tier or "write") or s.tiers["write"]
    model = (body.model or "").strip() or base_tier.model
    api_base = base_tier.api_base if body.api_base is None else (body.api_base.strip() or None)
    api_key = (body.api_key or "").strip() or base_tier.api_key
    if s.llm_fake:
        return {"ok": True, "fake": True, "model": llm.resolve_model(model), "latency_ms": 0,
                "reply": "离线演示模式下不会真正调用模型"}
    return llm.test_connection(model, api_base, api_key, timeout=min(base_tier.timeout, 30))


class TestSearchIn(BaseModel):
    source: str


@router.post("/test-search")
def test_search(body: TestSearchIn):
    if body.source not in _SOURCES:
        raise HTTPException(400, "未知来源")
    if get_settings().llm_fake:
        return {"ok": True, "fake": True, "latency_ms": 0, "sample": "离线演示模式下不会真正请求", "authenticated": False}
    return scholar.test_source(body.source)


class TestParserIn(BaseModel):
    parser: str  # grobid / mineru，使用已保存的地址


@router.post("/test-parser")
def test_parser(body: TestParserIn):
    from ..parsers import test_parser as probe

    s = get_settings()
    if body.parser == "grobid":
        return probe("grobid", s.grobid_url or "")
    if body.parser == "mineru":
        return probe("mineru", s.mineru_url or "")
    raise HTTPException(400, "未知解析器")
