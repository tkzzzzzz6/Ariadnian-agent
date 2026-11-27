# Gemma 4 E2B 本地推理与独立测评

入口位于 `experiments/ms_swift/gemma_eval/`，使用独立环境。CPU 准备阶段不导入
Torch、Transformers、MS-SWIFT 或应用后端；GPU 推理显式使用 `--execute`。
所有命令默认从 `experiments/ms_swift/` 执行。

固定模型为 `google/gemma-4-E2B` **Base**，revision
`433813a669639fd7eaa83155828e9000ed357d53`。原生 tokenizer 由既有模型锁校验。
当前协议使用 `BOS + 原始文本` completion，不套 IT 聊天模板，也不将官方 IT
模型的成绩当成 Base 基线。SFT 产物可以加载已绑定来源的 adapter；若后续另加
聊天模板评测，应建立新协议，所有对照组保持一致。

## 三类输入

| 类型 | 输入 | 输出与边界 |
|---|---|---|
| `profile` | UTF-8 种子文本 | 重复 token 构造精确 2K/4K/8K 输入；只测资源，不评分能力 |
| `benchmark` | 既有 `selected_records.jsonl` 与相邻 `fetch_manifest.json` | 复用冻结公开题目；生成逐题预测，交既有评分器离线评分 |
| `nll` | 带原始来源及 train/dev/test 划分的语料 | 只接受 dev/test，统计每个目标 token 的负对数似然 |

公开题完整提示词经过 Gemma 原生 tokenizer，预算为 **BOS + 正文 + 问题/指令 +
官方输出预留**。超长样本保留 `skipped_context_limit`，不偷偷截断或换题。
准备与执行都核验来源、请求文件和 tokenizer 哈希；执行前重新从来源编码，防止
修改请求及其哈希后错误地沿用旧计划。

NLL 每个窗口最多 `--context-length` 个 token，相邻窗口仅重叠一个前置 token；
每个正文/EOS 目标恰好计分一次，窗口首位为 `-100`，不加 padding，不跨窗口保留 KV。
这属于**窗口化 NLL**，不是全篇长上下文 NLL。跨模型比较须使用相同窗口协议。
部分文档失败时，整体 `mean_nll` 保持 `null`，另列成功文档条件均值和已评分 token 数。

## 准备计划（不加载权重）

```bash
uv run python -m gemma_eval prepare \
  --kind profile --input .cache/profile_seed.txt \
  --model-assets .cache/model_assets/gemma4-e2b \
  --cache-dir .cache/hf_hub \
  --lengths 2048 4096 8192 --max-new-tokens 32 \
  --output .cache/gemma_profile_new

uv run python -m gemma_eval prepare \
  --kind benchmark \
  --input ../../backend/publicbench/.cache/selected_v1/selected_records.jsonl \
  --model-assets .cache/model_assets/gemma4-e2b \
  --cache-dir .cache/hf_hub --context-length 8192 \
  --output .cache/gemma_public_new

uv run python -m gemma_eval prepare \
  --kind nll --split dev --context-length 2048 \
  --input ../../backend/longdoc_sources/.cache/pipeline_v1/corpus.jsonl \
  --model-assets .cache/model_assets/gemma4-e2b \
  --cache-dir .cache/hf_hub --output .cache/gemma_nll_new

uv run python -m gemma_eval run --plan .cache/gemma_profile_new/plan.json
```

输出目录必须为空或尚不存在。`--limit N` 仅选冻结输入的前 N 条，需在推理前决定；
不能按结果重选。评测计划引用绝对路径，迁移到服务器后须重新准备，不改旧清单冒充新计划。

2026-11-07 已在本地原始数据上生成
`.cache/gemma_eval_20261107_v1/`：

- `profile/plan.json`：精确 2,048 / 4,096 / 8,192 输入 token，分别预留 32 输出 token。
- `benchmark_8k/plan.json`：原 42 题全部保留；12 题符合完整 8K 预算，30 题超限。
- `nll_dev_2k/plan.json`：41 篇 dev 文档，324,973 个计划计分 token。

这些是**真实数据准备统计**，不是模型成绩或显存可用证明。

## 隔离推理环境

Python 3.12 环境使用 [requirements-inference.txt](../experiments/ms_swift/requirements-inference.txt)。
推荐将 venv 和模型缓存放在 `.cache/`，不向 `backend/.venv` 安装 GPU 依赖。

```bash
python3.12 -m venv .cache/inference_env
source .cache/inference_env/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-inference.txt
python -m pip install -e ../common -e .
python -m pip check
python -m pip freeze > .cache/inference_environment.txt
```

Windows 可用 `uv venv --python 3.12 .cache/inference_env` 创建环境，并将上面的
`python` 替换为 `.cache/inference_env/Scripts/python.exe`。Torch wheel、驱动和模型加载
是否正常须在该环境实际验证；存在某个版本不等于 CUDA 运行成功。

提前下载固定权重版本：

```bash
hf download google/gemma-4-E2B \
  --revision 433813a669639fd7eaa83155828e9000ed357d53 \
  --cache-dir .cache/hf_hub --include '*.json' '*.safetensors' '*.model'
```

执行过程使用 `local_files_only=True`，不会因缓存缺失自动联网换版本。运行时记录
`model.safetensors` 的 SHA-256，并对照包内 `model_weights.lock.json` 绑定的官方 LFS
内容 SHA-256 与大小。Hub/Xet 的内部缓存名称不一定是文件 SHA-256，因此不从缓存目录名
推断权重哈希；多层符号链接按最终文件内容核验。
加载前还会核对 HF 缓存的 `config.json`、`generation_config.json` 与模型锁的内容哈希，
并显式使用已锁定的模型配置。每次生成使用新建的固定贪心配置，不继承缓存或 adapter 的
采样、停止字符串、强制 token、最小输出长度及重复惩罚等设置。固定输出上限与 EOS 控制停止；
完整生效生成配置的哈希随运行记录保存。
每题复制独立 `GenerationConfig` 设置输出上限，避免在 `generate()` 中重复传入生成参数；
运行开始时记录本包源码、权重锁和推理依赖清单的哈希，保留该快照直到运行结束。

## 执行与资源指标

激活推理环境后直接运行其 Python，**不要用 CPU 准备环境的 `uv run` 执行 GPU 评测**。
推荐使用有明确总时限的入口，以下预算覆盖子进程加载和所有计划题目：

```bash
python -m gemma_eval.bounded \
  --plan .cache/gemma_eval_20261107_v1/profile/plan.json \
  --output .cache/gemma_profile_bounded_new --max-seconds 120 --execute
```

省略 `--execute` 只检查，不创建子进程或结果目录。`--max-seconds` 必须明确指定为
1–86,400 的整数；首轮建议小预算，失败或超时后先检查证据，再决定是否扩大。
包装器只启动当前 Python 的 `gemma_eval run`，不接受任意命令。Windows 超时通过
`taskkill /PID <自身创建的PID> /T /F` 请求清理该进程树，Linux 使用新建的进程组；
清理结果单独记录，失败不会冒充已经释放 GPU，也不会修改驱动策略。

原始逐题结果和清单保存在输出的 `run/`，stdout/stderr 日志保存在根目录。
根目录的 `predictions.jsonl` 与 `summary.json` 保留所有计划 ID：已完成结果原样保留，
准备阶段已确定的超限项保留 `skipped_context_limit`；超时首个尚未完成的可运行项标为
`interrupted_timeout`，其余可运行项为 `not_attempted_after_timeout`。
这些状态不是 OOM 或单题延迟测量；总时限可能耗在权重加载阶段。不要删除或改写子目录证据。
公开基准离线评分可使用根目录预测，以保留超时和未尝试的分母。

需要自行管理进程生命周期的服务器仍可使用原始 `python -m gemma_eval run` 入口，
但它没有独立进程层面的总超时限制。

默认 NF4、double quant、BF16 compute；可准备独立的 `--quantization bf16` 计划。
仅支持 CUDA 0，检查所有参数的设备标记为 CUDA，不启用框架 CPU/offload。NF4 面向支持的线性层，
不能把 Gemma 的 embedding/PLE 自动按四位估算。8GB 能否容纳须实测；加载失败同样留存。

`cpu_offload=false` 只表示上述框架检查，不能证明物理显存完全驻留，也不能排除 Windows
WDDM/驱动的系统内存分页。运行记录明确标注 `physical_vram_residency_verified=false`。
[NVIDIA NVML 文档](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)
说明 WDDM 下设备内存主要由 Windows 管理。当前未测量分页；长输入变慢本身不足以判定原因。

逐题记录：加载/生成分配峰值、保留显存峰值、首 token 时间、首 token 后解码速度、
端到端生成速度、实际生成 token、EOS/长度停止原因。CUDA 同步与 streamer 回调本身有开销；
首条为冷启动、不做预热，因此不得直接拿这些数字与其他推理引擎的宣传吞吐比较。
`input_tokens_per_first_token_second` 是输入长度除首 token 时间的近似指标，不是独立测得的
prefill kernel 吞吐。Torch allocator 指标不包含桌面及其他进程；整卡情况另看
`python -m cpt.profile inspect`。

每题后原子替换 `predictions.jsonl` 和进度清单。缺包、模型加载失败、OOM、超长、空回答
都保留原 ID 和分母；缺包记录包名与版本，OOM 尽可能记录当时 CUDA 显存。
原始异常文本、环境变量、凭据不写入汇总。中断时已有记录保留，不生成成功完成声明。

Adapter 使用 `prepare --adapter <checkpoint>` 绑定配置/权重哈希，并要求其原始
`launch_plan.json` 与成功退出的 `launch_receipt.json` 可验证、模型 revision 一致、数据计划仍有效。
没有来源链的外部 adapter 不在这个严格入口内；进程成功退出本身也不等于质量提升。

## 复用公开基准评分

切回 CPU 后端环境，从仓库根目录运行既有入口即可；该步骤不调用模型 API：

```bash
npm run eval:public -- rescore \
  --input publicbench/.cache/selected_v1/selected_records.jsonl \
  --predictions ../experiments/ms_swift/.cache/gemma_public_run_new/predictions.jsonl \
  --output evals/results/publicbench_gemma_new
```

如准备时用了 `--limit N`，评分也传相同 `--limit N`。评分器重新核验样本哈希，按任务的
官方指标评分，缺失/失败保留为失败；v1 与 v2 不混成一个“总体正确率”。原来的 DeepSeek
参考正文预算与这里的 Gemma 完整提示预算不同，不应直接解释为模型能力差异。

来源：[固定 Transformers Gemma 4 实现](https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/models/gemma4/modeling_gemma4.py)、
[Google Gemma 模型卡](https://huggingface.co/google/gemma-4-E2B)、
[项目公开基准协议](public-benchmarks.md)。
