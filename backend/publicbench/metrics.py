"""LongBench task-specific scores, adapted from the official MIT-licensed code.

Source: THUDM/LongBench at 2e00731f8d0bff23dc4325161044d0ed8af94c1e,
LongBench/metrics.py, LongBench/eval.py, pred.py and result.py.
Official score semantics are retained, including first-line task processing,
best-reference aggregation, Jieba Chinese tokenization, rouge==1.x and strict v2
answer extraction. Failures/skips remain zero-valued cases in the denominator.
"""
from __future__ import annotations

import importlib.metadata
import math
import re
import string
from collections import Counter, defaultdict

UPSTREAM_MIT_NOTICE = """MIT License

Copyright (c) 2023 THU-KEG & Zhipu AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

TASK_METRICS = {**{task: "qa_f1_score" for task in ("narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa")},
                "multifieldqa_zh": "qa_f1_zh_score", "dureader": "rouge_zh_score", "vcsum": "rouge_zh_score",
                **{task: "rouge_score" for task in ("gov_report", "qmsum", "multi_news", "samsum")},
                "trec": "classification_score", "lsht": "classification_score", "passage_count": "count_score",
                "passage_retrieval_en": "retrieval_score", "passage_retrieval_zh": "retrieval_zh_score",
                "lcc": "code_sim_score", "repobench-p": "code_sim_score"}
FIRST_LINE_TASKS = {"trec", "triviaqa", "samsum", "lsht"}
CN_PUNCTUATION = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."


def normalize_answer(text: str) -> str:
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def normalize_zh_answer(text: str) -> str:
    return "".join("".join(c for c in text.lower() if c not in set(string.punctuation + CN_PUNCTUATION)).split())


def f1_score(prediction: list[str], ground_truth: list[str], **kwargs) -> float:
    same = sum((Counter(prediction) & Counter(ground_truth)).values())
    if not same:
        return 0.0  # Official empty/empty is zero, unlike some generic QA scorers.
    precision, recall = same / len(prediction), same / len(ground_truth)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
    return f1_score(normalize_answer(prediction).split(), normalize_answer(ground_truth).split())


def _jieba_tokens(text: str) -> list[str]:
    import jieba
    jieba.setLogLevel(60)
    return list(jieba.cut(text, cut_all=False))


def qa_f1_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    predicted = [t for token in _jieba_tokens(prediction) if (t := normalize_zh_answer(token))]
    gold = [t for token in _jieba_tokens(ground_truth) if (t := normalize_zh_answer(token))]
    return f1_score(predicted, gold)


def rouge_score(prediction: str, ground_truth: str, **kwargs) -> float:
    try:
        from rouge import Rouge
    except ImportError as exc:
        raise RuntimeError("Official LongBench ROUGE requires the 'rouge' package (1.0.1); no surrogate score is substituted") from exc
    try:
        return Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"]
    except Exception:
        return 0.0  # The official implementation returns zero for empty/invalid text.


def rouge_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    return rouge_score(" ".join(_jieba_tokens(prediction)), " ".join(_jieba_tokens(ground_truth)))


def classification_score(prediction: str, ground_truth: str, **kwargs) -> float:
    matches = [name for name in kwargs["all_classes"] if name in prediction]
    # Preserve the official list-removal semantics, including prefix labels.
    for term in matches:
        if term in ground_truth and term != ground_truth:
            matches.remove(term)
    return 1.0 / len(matches) if ground_truth in matches else 0.0


def count_score(prediction: str, ground_truth: str, **kwargs) -> float:
    numbers = re.findall(r"\d+", prediction)
    return sum(number == str(ground_truth) for number in numbers) / len(numbers) if numbers else 0.0


def retrieval_score(prediction: str, ground_truth: str, **kwargs) -> float:
    ground_id = re.findall(r"Paragraph (\d+)", ground_truth)[0]
    return count_score(prediction, ground_id)


def retrieval_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    ground_id = re.findall(r"段落(\d+)", ground_truth)[0]
    return count_score(prediction, ground_id)


def code_sim_score(prediction: str, ground_truth: str, **kwargs) -> float:
    try:
        from fuzzywuzzy import fuzz
    except ImportError as exc:
        raise RuntimeError("Official code completion scoring requires optional fuzzywuzzy; no surrogate score is substituted") from exc
    line = next((line for line in prediction.lstrip("\n").split("\n") if "`" not in line and "#" not in line and "//" not in line), "")
    return fuzz.ratio(line, ground_truth) / 100


def extract_v2_answer(response: str) -> str | None:
    """Exact official pred.py extraction; a bare option letter is not accepted."""
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", response)
    return match.group(1) if match else None


def score_prediction(example: dict, prediction: str | dict | None, status: str = "ok") -> dict:
    if isinstance(prediction, dict):
        status = prediction.get("status", status)
        prediction = next((prediction[key] for key in ("answer", "prediction", "pred", "text", "output")
                           if isinstance(prediction.get(key), str)), None)
    benchmark = example["benchmark"]
    task = example["task"]
    metric = "accuracy_official_v2_extraction" if benchmark in {"v2", "longbench-v2"} else TASK_METRICS.get(task)
    if metric is None:
        raise ValueError(f"unsupported official metric for task {task}")
    result = {"id": example["id"], "benchmark": benchmark, "task": task,
              "domain": example.get("domain", task), "language": example.get("language", "unknown"),
              "metric": metric, "metric_name": metric, "score": 0.0, "score_percent": 0.0,
              "status": status, "official_answer_extraction": None}
    if status not in {"ok", "success"} or not isinstance(prediction, str) or not prediction.strip():
        if status in {"ok", "success"}:
            result["status"] = "empty_or_missing_prediction"
        return result
    references = example.get("reference_answers", example.get("answers"))
    if not isinstance(references, list) or not references or any(not isinstance(r, str) for r in references):
        raise ValueError("scoring requires the unchanged official nonempty reference list")
    if benchmark in {"v2", "longbench-v2"}:
        result["official_answer_extraction"] = extract_v2_answer(prediction)
        score = float(result["official_answer_extraction"] == references[0])
    else:
        if task in FIRST_LINE_TASKS:
            prediction = prediction.lstrip("\n").split("\n")[0]
        function = globals()[metric]
        classes = example.get("all_classes")
        if metric == "classification_score" and (not isinstance(classes, list) or not classes):
            raise ValueError("official classification scoring requires unchanged all_classes")
        score = max(function(prediction, reference, all_classes=classes) for reference in references)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("official metric returned a nonfinite/out-of-range score")
    result.update(status="ok", score=float(score), score_percent=100 * score)
    return result


def aggregate_scores(rows: list[dict]) -> dict:
    """Every provided sampled case, including failed/skipped cases, stays scored."""
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate prediction IDs in benchmark score aggregation")

    def group(items: list[dict]) -> dict:
        values = []
        successful = 0
        for row in items:
            ok = row.get("status") in {"ok", "success"}
            value = row.get("score", 0.0) if ok else 0.0
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("sample scores must be finite in [0,1]")
            values.append(float(value))
            successful += ok
        mean = sum(values) / len(values) if values else None
        return {"n": len(items), "successful": successful, "failed_or_missing": len(items) - successful,
                "coverage": successful / len(items) if items else 0.0,
                "mean_score": mean, "score_percent": 100 * mean if mean is not None else None}

    result = group(rows)
    for field in ("benchmark", "task", "domain", "language"):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row.get(field, "unknown"))].append(row)
        result["by_" + field] = {name: group(items) for name, items in sorted(groups.items())}
    for benchmark, summary in result["by_benchmark"].items():
        tasks = defaultdict(list)
        for row in rows:
            if row["benchmark"] == benchmark:
                tasks[row["task"]].append(row)
        means = [group(items)["mean_score"] for items in tasks.values()]
        summary["macro_task_score"] = sum(means) / len(means) if means else None
        summary["macro_task_score_percent"] = 100 * summary["macro_task_score"] if means else None
    if len(result["by_benchmark"]) > 1:
        result["mean_score"] = result["score_percent"] = None
    versions = {}
    for package in ("jieba", "rouge", "fuzzywuzzy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result.update(metric_implementation="adapted_official_LongBench_task_specific_v1_and_strict_v2",
                  dependency_versions=versions, failed_cases_score_zero=True, official_full_run=False,
                  note="Small-sample protocol; inspect task/benchmark scores. Cross-benchmark mean is intentionally omitted.")
    return result
