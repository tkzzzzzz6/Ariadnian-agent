"""LLM 接入层：底层用 LiteLLM，上层只暴露 chat / stream_chat / chat_json 三个函数。

- 按任务分四档模型（complete / write / judge / agent），见 config.py；
- 每档独立的超时、重试次数和备用模型（主模型失败时依次尝试 fallbacks）；
- 需要 JSON 的地方优先使用服务端 JSON 模式，不支持时退回到宽松解析；
- 每次调用都记录到 llm_calls 表（耗时、首 token 延迟、token 数、费用），/api/llm/stats 汇总；
- LLM_FAKE=1 时不调用任何模型，返回调用方给出的演示内容，便于离线开发和测试。

业务代码只依赖这三个函数，以后换成别的 SDK 也只需要改这一个文件。
"""
from __future__ import annotations

import json
import logging
import os
import re
import itertools
import time
from typing import Any, Callable, Iterator, Literal

from . import runstats, tracing
from .config import ModelTier, get_settings

os.environ.setdefault("LITELLM_LOG", "ERROR")
import litellm  # noqa: E402

litellm.drop_params = True  # 不同服务支持的参数不一样，不支持的直接丢弃而不是报错
litellm.suppress_debug_info = True
litellm.telemetry = False
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

_stream_ids = itertools.count(1)
Tier = Literal["complete", "write", "judge", "agent"]
FakeFn = Callable[[list[dict[str, str]]], str] | str | None


class LLMNotConfigured(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 模型名解析
# ---------------------------------------------------------------------------
_LEGACY_PREFIX = {
    "openai_like": "openai",
    "openai-compatible": "openai",
    "openai_compat": "openai",
    "custom": "openai",
    "openai": "openai",
    "openai_chat": "openai",
    "ollama": "ollama_chat",
    "deepseek": "deepseek",
}


_PROVIDERS = {str(getattr(p, "value", p)).lower() for p in litellm.provider_list}


def resolve_model(model: str) -> str:
    """补全 LiteLLM 的 provider 前缀。

    deepseek/deepseek-chat → 原样；mimo-v2.5-pro（旧配置）→ openai/mimo-v2.5-pro；
    Qwen/Qwen2.5-7B-Instruct（硅基流动这类带斜杠的模型名）→ openai/Qwen/Qwen2.5-7B-Instruct。
    """
    head = model.split("/", 1)[0].lower()
    if "/" in model and head in _PROVIDERS:
        return model
    prefix = _LEGACY_PREFIX.get(get_settings().provider, "openai")
    return f"{prefix}/{model}"


def tier_info() -> dict[str, dict[str, Any]]:
    s = get_settings()
    return {
        name: {"model": resolve_model(t.model), "api_base": t.api_base, "timeout": t.timeout,
               "retries": t.retries, "fallbacks": [resolve_model(m) for m in t.fallbacks]}
        for name, t in s.tiers.items()
    }


def _candidates(t: ModelTier) -> list[str]:
    return [resolve_model(t.model), *(resolve_model(m) for m in t.fallbacks)]


def _kwargs(t: ModelTier, model: str) -> dict[str, Any]:
    kw: dict[str, Any] = {"model": model, "timeout": t.timeout, "num_retries": t.retries}
    if t.api_key:
        kw["api_key"] = t.api_key
    if t.api_base:
        base = t.api_base.rstrip("/")
        # Ollama 原生接口不带 /v1；OpenAI 兼容接口需要 /v1
        if model.startswith(("ollama/", "ollama_chat/")) and base.endswith("/v1"):
            base = base[:-3]
        kw["api_base"] = base
    return kw


def _check_ready(t: ModelTier) -> None:
    s = get_settings()
    local = resolve_model(t.model).startswith(("ollama/", "ollama_chat/"))
    if not (t.api_key or local or s.provider == "ollama"):
        raise LLMNotConfigured("未配置模型 API Key（LLM_API_KEY / OPENAI_API_KEY），或设置 LLM_FAKE=1 离线演示")


# ---------------------------------------------------------------------------
# 调用日志
# ---------------------------------------------------------------------------
def _log(tier: str, model: str, *, stream: bool, ok: bool, latency: float, first_token: float | None = None,
         usage: Any = None, cost: float | None = None, error: str | None = None, stream_key: int | None = None) -> None:
    pt = getattr(usage, "prompt_tokens", None) if usage else None
    ct = getattr(usage, "completion_tokens", None) if usage else None
    cached = _cached_tokens(usage)
    reasoning_details = getattr(usage, "completion_tokens_details", None) if usage else None
    reasoning_tokens = (reasoning_details.get("reasoning_tokens") if isinstance(reasoning_details, dict)
                        else getattr(reasoning_details, "reasoning_tokens", None))
    tracing.annotate(model=model, tier=tier, prompt_tokens=pt, completion_tokens=ct, cached_tokens=cached,
                     cost_usd=cost, first_token_ms=int(first_token * 1000) if first_token is not None else None,
                     reasoning_tokens=reasoning_tokens, phase="finished", estimated=False)
    meter = runstats.current()
    if meter is not None:  # 属于某个 Agent 任务：计入这次运行的统计
        try:
            meter.record(model, ok=ok, latency=latency, first_token=first_token, prompt_tokens=pt,
                         completion_tokens=ct, cached_tokens=cached, cost=cost, stream_key=stream_key)
        except Exception:  # noqa: BLE001
            pass
    try:
        from .db import get_conn

        with get_conn() as conn:
            conn.execute(
                "INSERT INTO llm_calls (tier, model, stream, ok, latency_ms, first_token_ms, prompt_tokens,"
                " completion_tokens, cached_tokens, cost_usd, error, thread_id, run_id, span_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (tier, model, int(stream), int(ok), int(latency * 1000),
                 int(first_token * 1000) if first_token is not None else None, pt, ct, cached, cost,
                 tracing.scrub(error, 300) if error else None, meter.thread_id if meter else None, *tracing.ids()),
            )
    except Exception:  # 日志失败不能影响主流程
        pass


def _fake_record(tier: str, messages: list[dict[str, str]], text: str, started: float, first: float | None = None,
                 stream_key: int | None = None) -> None:
    """离线演示：按字数粗估 token，让 Agent 运行统计也有数可看（不写入 llm_calls）"""
    meter = runstats.current()
    if meter is None:
        return
    est = lambda s: max(1, int(len(s) / 1.6))  # noqa: E731  中英混排大约 1.6 字符一个 token
    configured = get_settings().tiers[tier].model  # 按设置里的模型算上下文窗口，演示更接近真实
    meter.record(resolve_model(configured) if configured else "fake", ok=True, latency=time.perf_counter() - started, first_token=first,
                 prompt_tokens=sum(est(str(m.get("content", ""))) for m in messages), completion_tokens=est(text),
                 cached_tokens=None, cost=None, stream_key=stream_key)
    tracing.annotate(model="离线演示", tier=tier, estimated=True, prompt_tokens=sum(est(str(m.get("content", ""))) for m in messages),
                     completion_tokens=est(text), phase="finished", reasoning_available=False)
    tracing.content(output=text)


def _cached_tokens(usage: Any) -> int | None:
    """命中缓存的输入 tokens：DeepSeek 用 prompt_cache_hit_tokens，OpenAI 兼容接口用 prompt_tokens_details.cached_tokens"""
    if not usage:
        return None
    v = getattr(usage, "prompt_cache_hit_tokens", None)
    if v is None:
        details = getattr(usage, "prompt_tokens_details", None)
        v = getattr(details, "cached_tokens", None) if details is not None else None
        if v is None and isinstance(details, dict):
            v = details.get("cached_tokens")
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _cost(resp: Any) -> float | None:
    try:
        return float(litellm.completion_cost(completion_response=resp))
    except Exception:
        return None


def _fake_text(messages: list[dict[str, str]], fake: FakeFn) -> str:
    if callable(fake):
        return fake(messages)
    if isinstance(fake, str):
        return fake
    last = messages[-1]["content"] if messages else ""
    return f"[离线演示输出] {last[:60]}"


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------
def _trace_messages(messages: list[dict]) -> list[dict]:
    # System prompts and protocol blocks are not diagnostic display content.
    return [{"role": m.get("role"), "content": str(m.get("content", ""))[:2000]}
            for m in messages if m.get("role") in ("user", "assistant")][-4:]


def _reasoning(message: Any) -> str | None:
    value = getattr(message, "reasoning_content", None)
    if isinstance(value, str) and value:
        return value
    blocks = getattr(message, "thinking_blocks", None)
    if isinstance(blocks, list):
        texts = [b.get("thinking", "") for b in blocks if isinstance(b, dict) and b.get("type") == "thinking"]
        return "\n".join(t for t in texts if isinstance(t, str)) or None
    return None


def _append_preview(previous: str, chunk: str, limit: int = 1800) -> str:
    """Keep the beginning and latest text without accumulating a whole stream."""
    text = previous + chunk
    if len(text) <= limit:
        return text
    marker = "\n…[中间内容已省略，以下为最新片段]…\n"
    return text[:500] + marker + text[-(limit - 500 - len(marker)):]


def _completion(kw: dict, tier: str, purpose: str = "模型调用", operation: str | None = None, attempt: int = 1):
    start = time.perf_counter()
    with tracing.span("llm", kw["model"], purpose, _trace_messages(kw["messages"])):
        tracing.annotate(model=kw["model"], tier=tier, phase="waiting", reasoning_available=False, operation=operation, attempt=attempt)
        try:
            resp = litellm.completion(**kw)
            message = resp.choices[0].message
            reasoning = _reasoning(message)
            tracing.annotate(reasoning_available=bool(reasoning))
            tracing.content(output=message.content, reasoning=reasoning)
            _log(tier, kw["model"], stream=False, ok=True, latency=time.perf_counter() - start,
                 usage=getattr(resp, "usage", None), cost=_cost(resp))
            return resp
        except Exception as e:
            _log(tier, kw["model"], stream=False, ok=False, latency=time.perf_counter() - start, error=repr(e))
            raise


def chat(
    messages: list[dict[str, str]],
    *,
    tier: Tier = "write",
    temperature: float = 0.3,
    max_tokens: int | None = None,
    json_mode: bool = False,
    fake: FakeFn = None,
) -> str:
    s = get_settings()
    if s.llm_fake:
        with tracing.span("llm", "fake", "模型调用 · 离线演示", _trace_messages(messages)):
            started = time.perf_counter()
            text = _fake_text(messages, fake)
            _fake_record(tier, messages, text, started)
            return text
    t = s.tiers[tier]
    _check_ready(t)
    last_err: Exception | None = None
    operation = f"request-{next(_stream_ids)}"
    attempt_number = 0

    def invoke(kw, purpose):
        nonlocal attempt_number
        attempt_number += 1
        return _completion(kw, tier, purpose, operation, attempt_number)

    for attempt, model in enumerate(_candidates(t)):
        kw = _kwargs(t, model) | {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        try:
            try:
                resp = invoke(kw, "模型调用" if attempt == 0 else "备用模型调用")
            except litellm.BadRequestError:
                if not json_mode:
                    raise
                kw.pop("response_format")
                resp = invoke(kw, "重试 · 兼容 JSON 输出")
            text = resp.choices[0].message.content or ""
            if not text.strip() and max_tokens and getattr(resp.choices[0], "finish_reason", None) == "length":
                kw["max_tokens"] = None
                resp = invoke(kw, "重试 · 正文为空且达到长度限制")
                text = resp.choices[0].message.content or ""
            return text
        except Exception as e:
            last_err = e
    raise last_err  # type: ignore[misc]


def stream_chat(
    messages: list[dict[str, str]],
    *,
    tier: Tier = "write",
    temperature: float = 0.3,
    fake: FakeFn = None,
) -> Iterator[str]:
    s = get_settings()
    meter = runstats.current()
    key = next(_stream_ids)
    if s.llm_fake:
        with tracing.span("llm", "fake", "模型调用 · 离线演示", _trace_messages(messages)):
            text = _fake_text(messages, fake)
            delay = float(os.getenv("LLM_FAKE_STREAM_DELAY", "0") or 0)
            start = time.perf_counter()
            for i in range(0, len(text), 6):
                if delay:
                    time.sleep(delay)
                if meter is not None:
                    meter.stream_tick(key, start)
                if i == 0:
                    tracing.annotate(phase="output", estimated=True, reasoning_available=False)
                tracing.activity()
                tracing.snapshot(output=text[:i + 6][:2001])
                yield text[i : i + 6]
            _fake_record(tier, messages, text, start, 0.0 if text else None, stream_key=key)
            return
    t = s.tiers[tier]
    _check_ready(t)
    last_err: Exception | None = None
    for attempt, model in enumerate(_candidates(t)):
        start = time.perf_counter()
        first: float | None = None
        first_reasoning: float | None = None
        usage = None
        stream = None
        output, reasoning_text = "", ""
        try:
            with tracing.span("llm", model, "模型调用" if attempt == 0 else "备用模型调用", _trace_messages(messages)) as trace:
                tracing.annotate(model=model, tier=tier, phase="waiting", reasoning_available=False, operation=f"stream-{key}", attempt=attempt + 1)
                try:
                    stream = litellm.completion(
                        **_kwargs(t, model), messages=messages, temperature=temperature, stream=True,
                        stream_options={"include_usage": True},
                    )
                    for chunk in stream:
                        tracing.activity()
                        if getattr(chunk, "usage", None):
                            usage = chunk.usage
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        reasoning = _reasoning(delta)
                        if reasoning:
                            if first_reasoning is None:
                                first_reasoning = time.perf_counter() - start
                                tracing.annotate(phase="reasoning", reasoning_available=True,
                                                 first_reasoning_ms=int(first_reasoning * 1000))
                            if trace and trace.recorder.capture:
                                reasoning_text = _append_preview(reasoning_text, reasoning)
                        text = getattr(delta, "content", None)
                        if text:
                            if first is None:
                                first = time.perf_counter() - start
                                tracing.annotate(phase="output")
                            if trace and trace.recorder.capture:
                                output = _append_preview(output, text)
                            if meter is not None:
                                meter.stream_tick(key, start)
                            yield text
                        tracing.snapshot(output=output or None, reasoning=reasoning_text or None)
                    _log(tier, model, stream=True, ok=True, latency=time.perf_counter() - start,
                         first_token=first, usage=usage, stream_key=key)
                except Exception as e:
                    _log(tier, model, stream=True, ok=False, latency=time.perf_counter() - start,
                         first_token=first, usage=usage, error=repr(e), stream_key=key)
                    raise
                finally:
                    tracing.content(output=output or None, reasoning=reasoning_text or None)
                    if stream is not None and callable(getattr(stream, "close", None)):
                        try:
                            stream.close()
                        except Exception:
                            pass
            return
        except Exception as e:
            if first is not None:
                raise
            last_err = e
    raise last_err  # type: ignore[misc]


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(text: str) -> Any:
    """宽松地从模型输出里解析 JSON（兼容 ```json 代码块和前后废话）。"""
    m = _JSON_BLOCK.search(text)
    candidate = (m.group(1) if m else text).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = candidate.find(open_c), candidate.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"模型输出不是合法 JSON：{text[:200]}")


def chat_json(messages: list[dict[str, str]], *, fake: FakeFn = None, retries: int = 1, **kw: Any) -> Any:
    msgs = list(messages)
    last_err: Exception | None = None
    for _ in range(retries + 1):
        text = chat(msgs, fake=fake, json_mode=True, **kw)
        try:
            return parse_json(text)
        except ValueError as e:
            last_err = e
            msgs = msgs + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "上面的输出不是合法 JSON，请只输出 JSON，不要任何解释。"},
            ]
    raise last_err  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def _pct(values: list[int], p: float) -> int | None:
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


def stats(days: int = 30) -> dict[str, Any]:
    """用量统计：各档汇总、按天趋势、按模型明细。费用按 pricing.py 的价格表（人民币）计算。"""
    from datetime import datetime, timedelta, timezone

    from . import pricing
    from .db import get_conn

    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM llm_calls WHERE ts >= datetime('now', ?) ORDER BY ts", (f"-{days} days",)
        )]
    custom = pricing.custom_prices()
    prices = {m: pricing.price_for(m, custom) for m in {r["model"] for r in rows}}
    for r in rows:
        r["cost_cny"] = pricing.call_cost(r, prices[r["model"]]) if r["ok"] else None
        # 按北京时间归到某一天
        try:
            t = datetime.fromisoformat(r["ts"].replace(" ", "T")).replace(tzinfo=timezone.utc)
            r["day"] = (t + timedelta(hours=8)).strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            r["day"] = None

    def agg(rs: list[dict]) -> dict[str, Any]:
        ok = [r for r in rs if r["ok"]]
        lat = [r["latency_ms"] for r in ok if r["latency_ms"] is not None]
        ftk = [r["first_token_ms"] for r in ok if r["first_token_ms"] is not None]
        costs = [r["cost_cny"] for r in ok if r["cost_cny"] is not None]
        return {
            "calls": len(rs),
            "errors": len(rs) - len(ok),
            "latency_p50_ms": _pct(lat, 0.5),
            "latency_p95_ms": _pct(lat, 0.95),
            "first_token_p50_ms": _pct(ftk, 0.5),
            "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in ok),
            "cached_tokens": sum(r.get("cached_tokens") or 0 for r in ok),
            "completion_tokens": sum(r["completion_tokens"] or 0 for r in ok),
            "cost_cny": round(sum(costs), 4) if costs else None,
            "unpriced_calls": sum(1 for r in ok if r["cost_cny"] is None),
            "cost_usd": round(sum(r["cost_usd"] or 0 for r in ok), 4),
        }

    tiers = {t: agg([r for r in rows if r["tier"] == t]) for t in ("complete", "write", "judge", "agent")}

    # 按天：补齐没有调用的日期，图表横轴连续
    today = (datetime.now(timezone.utc) + timedelta(hours=8)).date()
    span = min(days, 90)
    daily = []
    for i in range(span - 1, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        day_rows = [r for r in rows if r["day"] == d]
        entry: dict[str, Any] = {"date": d, "tiers": {}}
        for t in ("complete", "write", "judge", "agent"):
            rs = [r for r in day_rows if r["tier"] == t and r["ok"]]
            entry["tiers"][t] = {
                "calls": len(rs),
                "tokens": sum((r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0) for r in rs),
                "cost_cny": round(sum(r["cost_cny"] or 0 for r in rs), 6),
            }
        daily.append(entry)

    models = []
    for m in sorted(prices):
        a_ = agg([r for r in rows if r["model"] == m])
        p = prices[m]
        a_.update({"model": m, "price": None if p is None else {
            "input": p.input, "cached": p.cached, "output": p.output,
            "peak_multiplier": p.peak_multiplier, "source": p.source, "note": p.note}})
        models.append(a_)
    models.sort(key=lambda x: -(x["cost_cny"] or 0))

    return {
        "days": days, "tiers": tiers, "daily": daily, "models": models, "total": agg(rows),
        "currency": "CNY", "prices_checked": pricing.CHECKED, "pricing_pages": pricing.PRICING_PAGES,
    }


# ---------------------------------------------------------------------------
# 设置页「测试连接」
# ---------------------------------------------------------------------------
def test_connection(model: str, api_base: str | None, api_key: str | None, timeout: float = 30) -> dict[str, Any]:
    """用给定的（可能还没保存的）配置发一个极小的请求，返回耗时或错误信息。"""
    resolved = resolve_model(model)
    t = ModelTier(name="test", model=model, api_key=api_key, api_base=api_base, timeout=timeout, retries=0)
    kw = _kwargs(t, resolved)
    start = time.perf_counter()
    try:
        resp = litellm.completion(**kw, messages=[{"role": "user", "content": "Reply with the single word: OK"}],
                                  max_tokens=5, temperature=0)
        return {"ok": True, "model": resolved, "latency_ms": int((time.perf_counter() - start) * 1000),
                "reply": (resp.choices[0].message.content or "").strip()[:40]}
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if api_key:
            msg = msg.replace(api_key, "****")  # 报错信息里不回显 key
        return {"ok": False, "model": resolved, "latency_ms": int((time.perf_counter() - start) * 1000),
                "hint": _explain(e, msg), "error": f"{type(e).__name__}: {msg}"[:400]}


def _explain(e: Exception, msg: str) -> str:
    """把常见错误翻译成用户能直接处理的提示"""
    name = type(e).__name__
    low = msg.lower()
    if name == "AuthenticationError" or "authentication" in low or "401" in msg or "invalid api key" in low or "incorrect api key" in low:
        return "API Key 无效或没有权限，检查 key 是否填对、是否属于这个服务商"
    if name == "NotFoundError" or "model_not_found" in low or "does not exist" in low or "404" in msg:
        return "找不到这个模型，检查模型名（以服务商控制台为准）或接口地址"
    if name == "RateLimitError" or "429" in msg or "quota" in low or "insufficient" in low:
        return "被限流或余额不足，稍后再试或检查账户余额"
    if name in {"Timeout", "APITimeoutError"} or "timed out" in low or "timeout" in low:
        return "请求超时，检查网络或换一个更快的模型"
    if "connection" in low or name == "APIConnectionError":
        return "连不上接口地址，检查地址是否正确、网络或代理是否可用"
    if "llm provider not provided" in low or "provider" in low and "not" in low:
        return "模型名缺少服务商前缀，写成 openai/模型名 或 deepseek/模型名"
    return "调用失败，详细信息见下方"
