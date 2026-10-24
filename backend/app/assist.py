"""编辑器里的"快速路径"能力：行内补全、选中改写、找引用、引用核查、文献问答。

这些操作要求低延迟，直接调用 LLM，不经过 LangGraph。
"""
from __future__ import annotations

from . import tracing

import re
from functools import lru_cache
from typing import Iterator

from . import llm
from .config import ModelTier, get_settings
from .library import Paper, get_paper
from .retrieval import Hit, search

_CJK = re.compile(r"[一-鿿]")


def is_chinese(text: str) -> bool:
    return len(_CJK.findall(text)) > max(4, len(text) * 0.15)


@lru_cache(maxsize=512)
def _english_keywords(text: str) -> str:
    def fake(_: list) -> str:
        return "retrieval augmented generation large language models citation hallucination factual errors"

    try:
        return llm.chat(
            [
                {"role": "system", "content": "把用户给出的中文学术句子转换成 8-15 个英文学术检索关键词（含同义词和常见缩写），用空格分隔，只输出关键词。"},
                {"role": "user", "content": text[:600]},
            ],
            tier="judge", temperature=0, max_tokens=80, fake=fake,
        ).strip()
    except Exception:  # noqa: BLE001
        return ""


def retrieval_query(text: str) -> str:
    """跨语言检索：中文句子 + 英文关键词。文献库大多是英文论文，纯中文做 BM25 会检索不到。"""
    if is_chinese(text):
        kw = _english_keywords(text.strip())
        if kw:
            return f"{text}\n{kw}"
    return text


# ---------------------------------------------------------------------------
# 行内补全（Copilot 式灰色建议）
# ---------------------------------------------------------------------------
COMPLETE_SYSTEM = (
    "你是研究报告和学术论文的写作助手，负责在用户光标处给出自然的续写建议，类似代码编辑器里的 Copilot。\n"
    "规则：\n"
    "1. 只输出要插入光标处的文字，不要重复光标前已有的内容，不要解释，不要加引号。\n"
    "2. 与上下文语言一致（中文就写中文，英文就写英文），保持学术语体。\n"
    "3. 最多补全一句话（中文不超过 50 字，英文不超过 35 词）；如果光标在句子中间，就补完这句话。\n"
    "4. 不要编造具体数字、实验结果或引用；需要引用的地方宁可不写。\n"
    "5. 如果没有合适的续写，输出空字符串。"
)


def complete(before: str, after: str = "", title: str = "", section: str = "", use_library: bool = True) -> str:
    before_tail = before[-1500:]
    refs = ""
    if use_library and len(before_tail.strip()) > 20:
        hits = search(before_tail[-400:], top_k=2)
        if hits:
            refs = "\n".join(f"- {h.text[:400]}" for h in hits)
    user = (
        f"论文标题：{title or '（未命名）'}\n当前章节：{section or '（未知）'}\n"
        + (f"可参考的文献片段（只用来保证事实准确，不要照抄）：\n{refs}\n" if refs else "")
        + f"\n<光标前>\n{before_tail}\n</光标前>\n<光标后>\n{after[:300]}\n</光标后>\n\n请给出光标处的续写："
    )

    def fake(_: list) -> str:
        if is_chinese(before_tail):
            return "这一现象在近年来的相关研究中得到了广泛关注。"
        return " This observation has motivated a growing body of follow-up work."

    text = llm.chat(
        [{"role": "system", "content": COMPLETE_SYSTEM}, {"role": "user", "content": user}],
        tier="complete", temperature=0.4, max_tokens=120, fake=fake,
    )
    text = text.strip("\n").removeprefix("<光标处>").strip("`")
    # 模型偶尔会把光标前的最后几个字重复一遍，去掉
    for n in range(min(len(text), 30), 3, -1):
        if before_tail.endswith(text[:n]):
            text = text[n:]
            break
    # 英文续写时补一个空格
    if text and before_tail and not is_chinese(text) and before_tail[-1].isalnum() and text[0].isalnum():
        text = " " + text
    return text


# ---------------------------------------------------------------------------
# 选中文本改写
# ---------------------------------------------------------------------------
EDIT_ACTIONS: dict[str, str] = {
    "polish": "润色这段文字：修正语法和用词，让表达更流畅、更符合学术写作规范，但不要改变原意和信息量。",
    "academic": "把这段文字改写成更正式的学术语体，避免口语化表达。",
    "rewrite": "换一种表达方式重写这段文字，保持原意，避免与原文句式雷同。",
    "expand": "在不编造事实和数据的前提下扩写这段文字，补充论证逻辑、过渡和解释，长度约为原文的 1.5–2 倍。",
    "shorten": "精简这段文字，删去冗余表达，保留全部关键信息，长度约为原文的 60%。",
    "translate_en": "把这段文字翻译成地道的学术英文。",
    "translate_zh": "把这段文字翻译成规范的学术中文。",
    "custom": "",
}


def edit_stream(text: str, action: str, instruction: str = "", context: str = "") -> Iterator[str]:
    task = EDIT_ACTIONS.get(action)
    if task is None:
        raise ValueError(f"未知操作：{action}")
    if action == "custom":
        if not instruction.strip():
            raise ValueError("自定义改写需要填写要求")
        task = instruction.strip()
    elif instruction.strip():
        task += f"\n额外要求：{instruction.strip()}"
    system = (
        "你是严谨的学术写作编辑。只输出修改后的文字本身，不要解释、不要加引号或 markdown 代码块。"
        "保留原文中的引用标记（如 [1]、[@key]、\\cite{...}）和公式，不要新增引用。"
    )
    user = (f"上下文（仅供参考，不要输出）：\n{context[-800:]}\n\n" if context.strip() else "") + (
        f"任务：{task}\n\n需要处理的文字：\n{text}"
    )

    def fake(_: list) -> str:
        if action == "translate_en":
            return "This paragraph has been translated into academic English (offline demo)."
        if action == "translate_zh":
            return "这段文字已被翻译为学术中文（离线演示）。"
        if action == "shorten":
            return text[: max(10, int(len(text) * 0.6))]
        if action == "expand":
            return text + ("此外，该结论也与已有研究的观察相一致，进一步说明了方法的有效性。" if is_chinese(text)
                           else " Moreover, this is consistent with prior observations, further supporting the approach.")
        return re.sub(r"非常|很", "较为", text) if is_chinese(text) else text.replace("very ", "").replace("a lot of", "substantial")

    yield from llm.stream_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}], tier="write", temperature=0.3, fake=fake
    )


# ---------------------------------------------------------------------------
# 为一句话找引用 / 引用核查
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = (
    "你是严格的学术审稿人。判断给定的文献片段能否支撑论文中的这句话。"
    '只输出 JSON：{"verdict": "supports" | "partial" | "unrelated", "reason": "不超过40字的中文理由"}。'
    "supports=片段直接支持该论断；partial=相关但只支持一部分或需要更多证据；unrelated=不能支撑。"
)


@lru_cache(maxsize=1024)
def _cached_support(sentence: str, paper_id: str, title: str, year: int | None, snippet: str,
                    tier_config: ModelTier, fake_mode: bool, provider: str, prompt: str) -> dict:
    # Configuration and evidence content participate in the key. Failed calls
    # raise out of the cached function, so transient failures are never cached.
    user = f"论文中的句子：{sentence}\n\n文献：{title} ({year or 'n.d.'})\n片段：{snippet}"

    def fake(_: list) -> str:
        from .retrieval import tokenize

        overlap = len(set(tokenize(retrieval_query(sentence))) & set(tokenize(snippet)))
        verdict = "supports" if overlap >= 3 else ("partial" if overlap >= 1 else "unrelated")
        return '{"verdict": "%s", "reason": "离线演示：按关键词重合度(%d)判断"}' % (verdict, overlap)

    data = llm.chat_json(
        [{"role": "system", "content": prompt}, {"role": "user", "content": user}],
        tier="judge", temperature=0, fake=fake,
    )
    verdict = data.get("verdict", "partial")
    if verdict not in {"supports", "partial", "unrelated"}:
        verdict = "partial"
    return {"verdict": verdict, "reason": str(data.get("reason", ""))[:120]}


def judge_support(sentence: str, paper: Paper, snippet: str) -> dict:
    try:
        settings = get_settings()
        return dict(_cached_support(sentence.strip(), paper.id, paper.title, paper.year, snippet[:1500],
                                    settings.tiers["judge"], settings.llm_fake, settings.provider, JUDGE_SYSTEM))
    except Exception as e:  # noqa: BLE001
        return {"verdict": "unknown", "reason": f"判断失败：{e}"}


def _best_hit_per_paper(hits: list[Hit]) -> list[Hit]:
    seen: dict[str, Hit] = {}
    for h in hits:
        if h.paper_id not in seen:
            seen[h.paper_id] = h
    return list(seen.values())


def suggest_citations(sentence: str, k: int = 5, judge: bool = True) -> list[dict]:
    hits = _best_hit_per_paper(search(retrieval_query(sentence), top_k=k * 3))[:k]
    out = []
    for h in hits:
        paper = get_paper(h.paper_id)
        if not paper:
            continue
        item = {
            "paper": paper.model_dump(),
            "snippet": h.text[:600],
            "section": h.section,
            "page": h.page,
            "score": h.score,
        }
        if judge:
            # Use the same per-paper evidence selection and verdict as the
            # post-insertion verifier, rather than judging the global-search hit.
            item.update(verify_citation(sentence, paper.id))
        out.append(item)
    if judge:
        rank = {"supports": 0, "partial": 1, "unknown": 2, "unrelated": 3}
        out.sort(key=lambda x: rank.get(x.get("verdict", "unknown"), 2))
    return out


def verify_citation(sentence: str, paper_id: str) -> dict:
    paper = get_paper(paper_id)
    if not paper:
        return {"paper_id": paper_id, "verdict": "missing", "reason": "文献库中没有这篇文献"}
    hits = [h for h in search(retrieval_query(sentence), top_k=1, paper_ids=[paper_id]) if h.paper_id == paper_id]
    if not hits:
        return {"paper_id": paper_id, "verdict": "unknown", "reason": "未检索到足够依据，暂时无法判断是否支持", "snippet": ""}
    res = judge_support(sentence, paper, hits[0].text)
    return {"paper_id": paper_id, "snippet": hits[0].text[:600], "page": hits[0].page, "section": hits[0].section, **res}


# ---------------------------------------------------------------------------
# 基于文献库的问答
# ---------------------------------------------------------------------------
ASK_SYSTEM = (
    "你是研究报告写作助手，根据用户文献库中检索到的片段回答问题。"
    "每个关键论断后用 [n] 标注来源编号（对应给出的片段编号）；片段里没有的信息要明确说明“文献库中未找到”，不要编造。"
    "使用与问题相同的语言回答，简洁、有条理。"
)


def ask_prepare(question: str, k: int = 6) -> tuple[list[dict], list[dict[str, str]]]:
    hits = search(retrieval_query(question), top_k=k)
    sources = []
    blocks = []
    for i, h in enumerate(hits, start=1):
        p = get_paper(h.paper_id)
        if not p:
            continue
        sources.append({"n": i, "paper_id": p.id, "bibkey": p.bibkey, "title": p.title, "year": p.year,
                        "section": h.section, "page": h.page, "snippet": h.text[:400]})
        blocks.append(f"[{i}] 《{p.title}》({p.year or 'n.d.'}) {h.section or ''} p.{h.page or '-'}\n{h.text}")
    context = "\n\n".join(blocks) if blocks else "（文献库为空或没有检索到相关内容）"
    messages = [
        {"role": "system", "content": ASK_SYSTEM},
        {"role": "user", "content": f"检索到的片段：\n{context}\n\n问题：{question}"},
    ]
    return sources, messages


def ask_stream(question: str, k: int = 6) -> tuple[list[dict], Iterator[str]]:
    sources, messages = ask_prepare(question, k)
    fake = (
        f"根据文献库，与“{question[:30]}”最相关的是《{sources[0]['title']}》[1]。（离线演示回答）"
        if sources else "文献库中未找到相关内容。（离线演示回答）"
    )
    return sources, llm.stream_chat(messages, tier="write", fake=fake)


# ---------------------------------------------------------------------------
# 页边批注：对一个段落做引用核查、缺引用检测和精简建议
# ---------------------------------------------------------------------------
# 引用标记（pandoc 语法）：[@key]、[@a; @b]，可带页码 [@key, p. 3]
_CITE_ITEM = r"@[\w:.-]+(?:,\s*[^;\[\]@]{1,30})?"
CITE_TOKEN = re.compile(r"\[((?:" + _CITE_ITEM + r"(?:\s*;\s*)?)+)\]")


def cite_keys(token_body: str) -> list[str]:
    """[@a, p. 3; @b] 里的 bibkey：['a', 'b']（去掉页码等定位信息）"""
    return [k.split(",")[0].strip().lstrip("@") for k in token_body.split(";") if k.strip()]
# 英文分号不作为句子边界：[@a; @b] 这样的引用标记里也有分号
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；])|(?<=[.])\s+")
_WORDY = re.compile(r"非常|在很大程度上|一定程度上|总的来说|众所周知|\bvery\b|\ba lot of\b|\bin order to\b", re.I)


def split_sentences(paragraph: str) -> list[str]:
    # 引用标记里可能有 "p. 3" 这样的句点，先换成占位符再分句，避免在引用中间断开
    tokens: list[str] = []

    def hide(m: re.Match) -> str:
        tokens.append(m.group(0))
        return f"\x00{len(tokens) - 1}\x00"

    def show(t: str) -> str:
        return re.sub(r"\x00(\d+)\x00", lambda m: tokens[int(m.group(1))], t)

    parts = [p for p in _SENT_SPLIT.split(CITE_TOKEN.sub(hide, paragraph)) if p and p.strip()]
    out: list[str] = []
    for p in parts:
        # "……结论。[@key]" —— 句末标点之后的引用属于前一句
        m = re.match(r"^\s*((?:\x00\d+\x00\s*)+)(.*)$", p, re.S)
        if m and out:
            out[-1] += m.group(1).rstrip()
            p = m.group(2)
        if p.strip():
            out.append(p)
    return [show(p) for p in out]


def strip_cites(text: str) -> str:
    return CITE_TOKEN.sub("", text)


REVIEW_SYSTEM = (
    "你是严谨的学术论文审稿人，帮助作者改进一个段落。只输出 JSON，不要解释。"
)


def _review(paragraph_marked: str, plain: str, title: str) -> dict:
    """让模型找：缺引用的事实性论断（≤2）+ 值得精简的一句（≤1）。"""
    user = (
        f"论文题目：{title or '（未命名）'}\n"
        f"段落（[引用] 表示该处已有引用）：\n{paragraph_marked}\n\n"
        "请找出：\n"
        "1. 需要文献支撑但没有引用的事实性论断，最多 2 句。必须原样摘录原文中的完整句子；"
        "常识、作者对自己方法的描述、过渡句都不算。\n"
        "2. 最多 1 句表达明显冗余或口语化、值得精简的句子，给出改写。只改表达、不改意思，"
        "不要选含 [引用] 的句子。\n"
        "没有就返回空数组 / null。输出格式：\n"
        '{"missing": [{"sentence": "原句", "reason": "不超过20字"}], '
        '"rewrite": {"sentence": "原句", "revised": "改写", "reason": "不超过20字"} 或 null}'
    )

    def fake(_: list) -> str:
        import json as _json

        sents = [s.strip() for s in split_sentences(plain)]
        wordy = next((s for s in sents if _WORDY.search(s)), None)
        marked_sents = split_sentences(paragraph_marked.replace("[引用]", "[@existing]"))
        missing = [s.strip() for s in marked_sents if len(s.strip()) >= 15 and s.strip() != wordy and not CITE_TOKEN.search(s)][:1]
        data: dict = {"missing": [{"sentence": s, "reason": "事实性论断缺少文献支撑（离线演示）"} for s in missing], "rewrite": None}
        if wordy:
            data["rewrite"] = {"sentence": wordy, "revised": _WORDY.sub("", wordy).replace("  ", " "), "reason": "删去冗余修饰（离线演示）"}
        return _json.dumps(data, ensure_ascii=False)

    return llm.chat_json(
        [{"role": "system", "content": REVIEW_SYSTEM}, {"role": "user", "content": user}],
        tier="judge", temperature=0, fake=fake,
    )


@tracing.traced("tool", "检查引用与生成批注")
def annotate(paragraph: str, title: str = "", check_citations: bool = True, review: bool = True) -> list[dict]:
    """paragraph 使用 [@bibkey] 表示引用。返回的 anchor 都是去掉引用标记后的原文片段，前端据此定位。"""
    import hashlib

    from .library import list_papers

    by_key = {p.bibkey: p for p in list_papers()}
    sents = split_sentences(paragraph)
    anns: list[dict] = []

    def aid(kind: str, anchor: str) -> str:
        return hashlib.md5(f"{kind}:{anchor}".encode()).hexdigest()[:10]

    # 1) 已有引用 → 核查原文是否支撑
    if check_citations:
        for s in sents:
            keys = [k for m in CITE_TOKEN.finditer(s) for k in cite_keys(m.group(1))]
            if not keys:
                continue
            anchor = strip_cites(s).strip()
            items = []
            for k in dict.fromkeys(keys):
                p = by_key.get(k)
                if not p:
                    items.append({"paper_id": None, "bibkey": k, "title": k, "verdict": "missing", "reason": "文献库中没有这篇文献"})
                    continue
                r = verify_citation(anchor, p.id)
                items.append({"paper_id": p.id, "bibkey": k, "title": p.title, "verdict": r["verdict"],
                              "reason": r.get("reason", ""), "snippet": r.get("snippet", ""), "page": r.get("page")})
            anns.append({"id": aid("cite", anchor), "kind": "cite_check", "anchor": anchor, "items": items})

    # 2) 缺引用 + 精简建议
    plain = strip_cites(paragraph)
    if review and len(plain.strip()) >= 30:
        marked = CITE_TOKEN.sub("[引用]", paragraph)
        try:
            data = _review(marked, plain, title)
        except Exception:  # noqa: BLE001  审稿失败不影响引用核查结果
            data = {}
        contexts = [(strip_cites(s).strip(), bool(CITE_TOKEN.search(s))) for s in sents]
        seen_missing: set[str] = set()
        for m in (data.get("missing") or [])[:2]:
            sent = strip_cites(str(m.get("sentence", ""))).replace("[引用]", "").strip()
            if len(sent) < 6 or sent not in plain:
                continue  # 模型没有原样摘录 → 无法定位，丢弃
            matches = [(text, cited) for text, cited in contexts if sent in text]
            if len(matches) != 1 or matches[0][1]:
                continue  # 已有引用的句子走引用核查；重复原句无法可靠定位
            sent = matches[0][0]  # 候选与复核均判断完整句子，不只判断模型摘出的半句
            if sent in seen_missing:
                continue
            seen_missing.add(sent)
            cands = [c for c in suggest_citations(sent, k=3, judge=True) if c.get("verdict") in {"supports", "partial"}][:2]
            anns.append({
                "id": aid("missing", sent), "kind": "missing_citation", "anchor": sent,
                "reason": str(m.get("reason", ""))[:60],
                "candidates": [{"paper_id": c["paper"]["id"], "bibkey": c["paper"]["bibkey"], "title": c["paper"]["title"],
                                "year": c["paper"].get("year"), "verdict": c["verdict"], "reason": c.get("reason", ""),
                                "snippet": c["snippet"][:300], "page": c.get("page")} for c in cands],
            })
        rw = data.get("rewrite") or None
        if isinstance(rw, dict):
            orig = str(rw.get("sentence", "")).strip()
            revised = str(rw.get("revised", "")).strip()
            matches = [(text, cited) for text, cited in contexts if orig and orig in text]
            if (orig and revised and orig != revised and orig in plain and "[引用]" not in orig
                    and len(matches) == 1 and not matches[0][1] and not CITE_TOKEN.search(revised) and "[引用]" not in revised):
                anns.append({"id": aid("rewrite", orig), "kind": "rewrite", "anchor": orig, "revised": revised,
                             "reason": str(rw.get("reason", ""))[:60]})
    return anns
