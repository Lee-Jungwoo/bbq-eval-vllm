"""BBQ evaluation with vLLM first-token logprobs.

Modules, by pipeline stage:

  data loading, prompt building   pipeline.py (original, 3 options)
                                  pipeline_binary.py (forced choice, 2 options)
                                  tokenizer_checker.py (label-token choice)
  inference (GPU server)          serve_logprobs.py  (standalone; vllm.LLM offline)
  scoring                         pipeline.py / pipeline_binary.py  `collect`
  aggregation                     aggregate.py, aggregate_binary.py
  metric definitions, publishing  bbq_metrics.py, publish.py

Run modules with `python -m bbq_eval.<module>` and `src/` on PYTHONPATH, or use the
entry points under `scripts/`. serve_logprobs.py imports nothing from this package
so it can be copied to the GPU server on its own.
"""
