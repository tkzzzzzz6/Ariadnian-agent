"""只展示聚合准备证据，公开页面不包含逐题问答。"""
import argparse
import html
import json
from pathlib import Path


def bars(counts):
    width, top = 560, max(counts.values(), default=1) or 1
    body = ''
    for i, (label, value) in enumerate(counts.items()):
        y = i * 38
        body += (f'<text x="0" y="{y+19}">{html.escape(str(label))}</text>'
                 f'<rect x="210" y="{y+4}" width="{280*value/top:.2f}" height="22" rx="3" fill="#247ba0"/>'
                 f'<text x="505" y="{y+20}">{value}</text>')
    return f'<svg role="img" aria-label="样本数分布" viewBox="0 0 {width} {max(38, len(counts)*38)}">{body}</svg>'


def render_report(data, partition):
    h = html.escape
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
             '<title>Issue 15 · 数据与评测准备</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#f4f7fa;color:#19334a;max-width:1100px;margin:32px auto;padding:0 20px}section{background:white;padding:24px;margin:18px 0;border-radius:10px;border:1px solid #d6e2e9}svg{width:100%;max-width:600px;font-size:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere}strong{color:#974914}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #d6e2e9;text-align:left}</style>',
             '<h1>下一阶段：数据与评测准备</h1><p><strong>不是能力提升结果：未执行新训练、Base 校准或确认测评。</strong></p>',
             '<section><h2>长短配对候选</h2><p>同问题、答案、来源与 split；长臂添加真实不同来源块，保留原顺序。</p>',
             bars(data['counts']),
             f"<p>{data['source_documents']} 篇来源；连接来源族 {h(str(data['connected_families']))}；跨文档来源族 {h(str(data['crossdoc_families']))}。</p>",
             '<p>每连接来源族最多4题。候选标签及字符串证据检查不等于人工验证，未获新增内容准入。</p></section>']
    for split, counts in data['languages_by_split'].items():
        parts += [f'<section><h2>{h(split)} 语言</h2>', bars(counts), '</section>']
    for split, counts in data['directions'].items():
        parts += [f'<section><h2>{h(split)} 预期任务方向</h2>', bars(counts),
                  '<p>多证据汇总与比较不自动标为多跳；方向未独立验证，比例未配平。</p></section>']
    parts += ['<section><h2>原生内容长度</h2><table><tr><th>条件</th><th>最小</th><th>最大</th></tr>']
    for arm, (low, high) in data['token_ranges'].items():
        parts += [f'<tr><td>{h(arm)}</td><td>{low}</td><td>{high}</td></tr>']
    parts += ['</table><p>仅消息内容 token；完整 SWIFT 模板仍待实际核验。长短输入与 FLOPs 未配平。</p></section>',
              '<section><h2>独立评测分区</h2>', bars(partition['selected_counts']),
              '<p>目标：开发40、示范2、确认160。实际按来源容量冻结；旧250不变，旧RULER仅作回归。</p>',
              '<h3>未补齐的来源缺额</h3><pre>' + h(json.dumps(partition['shortfalls'], ensure_ascii=False, indent=2)) + '</pre>',
              '<p>主 LongBench 输入至少8192，输入加官方输出不超过16384，不截断。缺失任务不支持泛化到该任务的结论。</p></section>',
              '<section><h2>下一道门</h2><ol><li>人工审阅新增问答、来源与许可；原40条仅在哈希仍相同时沿用旧批准。</li>',
              '<li>固定 Base / bf16 / 无 adapter，执行24道 BoolQ ×4配置，90%格式门槛及无失败/空输出；按固定优先级选择。</li>',
              '<li>重新核对最终训练来源，绑定真实模型/计划/逐题哈希；所有臂沿用同一配置，再封存确认协议。</li>',
              '<li>单 seed 配对试验后再考虑多 seed；CPT、公开数据发布、8GB实测及跨VM完整恢复尚未在本轮执行。</li></ol></section></html>']
    return '\n'.join(parts) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--aggregate', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    data = json.loads((args.aggregate / 'data_summary.json').read_text(encoding='utf-8'))
    partition = json.loads((args.aggregate / 'partition_summary.json').read_text(encoding='utf-8'))
    args.output.write_text(render_report(data, partition), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
