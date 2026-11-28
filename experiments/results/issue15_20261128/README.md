# Issue 15：下一阶段本地准备

[可视化报告](report.html)展示实际候选与评测分区；[操作说明](../../../docs/gemma-eval-calibration-data-v2.md)列出冻结规则与复现命令。

完成128train/32dev长短配对候选，48篇来源、40个连接来源族，中文76组。短/长消息内容长度为226–1103 /4394–7463原生token，完整SWIFT模板尚未实测。新增120组与36篇来源待实际人工审核；原40组/12篇来源已确认哈希一致，保留旧内部用途批准。没有产生新的训练批准或training export。

独立评测冻结开发39、示范2、确认候选150。NarrativeQA只剩1个独立来源组，开发少1题、确认少10题，未用重复来源补齐。主长文输入至少8K，总预算不超过16K；旧250和严格评分代码未修改。

- `data_summary.json`、`data_source.lock.json`：实际候选覆盖与来源指纹。
- `admission_status_summary.json`：旧批准兼容性与新增内容待审范围；不授予新准入。
- `partition_summary.json`、`partition_source.lock.json`：完整池隔离、实际数量、短缺与冻结350篇来源范围。
- `final_sources_binding_validation.json`：最终48篇来源与隔离范围逐篇匹配；确认协议尚未封存。
- `execution_status_summary.json`：CPU计划96条ready；新GPU、模板实测、校准、SFT、确认均未执行。
- `material_archive_manifest.json`：私人备份的字节数、SHA与云端读回状态。

本报告是数据准备证据，不是训练收益或正式长文泛化结果。原始问答、预测、语料、tokenizer与权重不进Git。本轮保留既有CPT和公开HF发布边界。

完整材料已备份到[私人Drive目录](https://drive.google.com/drive/folders/1FGW1orPYmm-lZqJerijGVA3sLvSeDoEg)，v3 ZIP云端下载读回的11,170,263字节及SHA256已核对。v1/v2保留为准备历史，最终以v3为准。
