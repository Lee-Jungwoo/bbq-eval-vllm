# vLLM 서빙 설정

이 문서는 BBQ 평가에서 모델을 vLLM으로 어떻게 띄웠는지, 어떤 값이 어디에 있는지,
그중 무엇이 실행 기록으로 확인되고 무엇이 확인되지 않는지를 정리한다.

## 실행 방식

OpenAI 호환 HTTP 서버를 띄우지 않았다. `src/bbq_eval/serve_logprobs.py`가 vLLM의
오프라인 `LLM` 클래스로 엔진을 만들고, 미리 토큰화된 프롬프트를 묶음 단위로
`llm.generate()`에 넘겨 **첫 출력 토큰의 logprob**만 받아 온다. 답 문자(A, B, C)의
logprob 중 가장 큰 것을 예측으로 쓰므로 디코딩 결과 텍스트는 쓰지 않는다.

작업은 로컬과 GPU 서버로 나뉜다.

1. 로컬 `prepare`: 데이터 로딩, 프롬프트 구성, chat template 적용, 토큰화까지 끝내고
   `requests.jsonl`(헤더 1줄과 `{"index", "prompt_token_ids"}` 줄들)을 만든다.
2. GPU 서버: `serve_logprobs.py` 한 파일만 복사해 실행한다. 이 파일은 BBQ를 모르고
   토큰 id를 받아 logprob을 돌려주는 일만 한다. 결과는 `responses.jsonl`에 묶음마다
   flush되며, 다시 실행하면 이미 있는 index는 건너뛴다(중단 후 재개 가능).
3. 로컬 `collect`, `aggregate`: 응답을 문항 정보와 합쳐 채점하고 집계한다.

토큰화를 로컬에서 한 번만 하는 이유는 Mistral 계열처럼 chat template이 jinja 문자열이
아닌 토크나이저에서, 렌더링된 텍스트를 서버에서 다시 인코딩하면 `[INST]` 같은 제어
토큰이 일반 문자로 바뀌기 때문이다(`pipeline.py`의 `REQUEST_SCHEMA` 주석).

## 설정 파일

모델마다 `configs/serving/<model>.yaml`이 하나씩 있다(24개). `scripts/serve.py`가 이
파일을 읽어 `serve_logprobs.py` 인자로 바꿔 실행한다. 값은 실험을 실제로 돌린
`scripts/run_ladder.sh`의 env 기본값과 `serve_logprobs.py`의 argparse 기본값을 그대로
옮긴 것이다.

```bash
# 준비된 run 디렉터리를 입력으로 (revision은 그 run의 meta.json 값을 쓴다)
python scripts/serve.py --config configs/serving/qwen2.5-3b-instruct.yaml \
    --run-dir results/runs/Qwen2.5-3B-Instruct_bfloat16_6db43b09

# 실제 실행 없이 명령과 환경 변수만 출력
python scripts/serve.py --config configs/serving/qwen2.5-3b-instruct.yaml \
    --run-dir results/runs/Qwen2.5-3B-Instruct_bfloat16_6db43b09 --dry-run
```

## 엔진 인자 (`vllm.LLM(...)`)

| 인자 | 값 | 어디서 오는가 |
|---|---|---|
| `model`, `revision` | HF 모델 이름, 고정 commit sha | `prepare`가 `main`을 sha로 바꿔 `meta.json`에 기록 |
| `dtype` | `bfloat16` (24개 모델 전부) | `run_ladder.sh` `DTYPE`, 모든 run의 `meta.json` |
| `max_model_len` | 1024 | `run_ladder.sh` `MAX_MODEL_LEN`. 길이가 기록된 run 중 가장 긴 프롬프트는 279 토큰(gpt-oss-20b, 원 조건) |
| `gpu_memory_utilization` | 0.90 | `run_ladder.sh` `GPU_MEM_UTIL` |
| `tensor_parallel_size` | GPU 목록의 장치 수 (기본 `GPU=1`이라 1) | `run_ladder.sh` `TP` |
| `max_logprobs` | 50 | `serve_logprobs.py --top-logprobs` 기본값 |
| `max_num_seqs` | 지정 안 함 (vLLM 기본값) | `run_ladder.sh` `MAX_NUM_SEQS`, 기본은 빈 값 |
| `limit_mm_per_prompt` | Ministral 3종만 `{"image": 0}` | `run_ladder.sh` 사용 예시와 `docs/pipeline.md` |
| `quantization` | 넘기지 않음 | `serve_logprobs.py`에 해당 인자가 없음 |

`batch_size`(기본 512)는 vLLM 인자가 아니라 `serve_logprobs.py`가 `llm.generate()`
한 번에 넘기는 프롬프트 수다. 실제로 동시에 스케줄되는 시퀀스 수는 vLLM 스케줄러의
`max_num_seqs`가 정하고, 이 값은 모든 run에서 vLLM 기본값으로 두었다. `run_ladder.sh`에
`MAX_NUM_SEQS`와 `TP` 옵션이 추가된 것은 Qwen2.5-14B를 돌리다 메모리가 부족했을 때다
(커밋 74b79f3, 64ddd15). 그 모델에 실제로 어떤 값을 썼는지는 기록이 없다.

## Sampling 설정 (`SamplingParams`)

| 항목 | 값 | 이유 |
|---|---|---|
| `max_tokens` | 1 | 첫 토큰 logprob만 쓴다 |
| `temperature` | 1.0 | 반환되는 logprob이 온도 보정 없는 log-softmax가 되도록. 샘플된 토큰은 쓰지 않는다 |
| `top_p` | 1.0 | 같은 이유 |
| `logprobs` | 50 | 답 문자 토큰이 상위 50개 밖이면 null로 기록한다 |
| `seed` | 지정 안 함 | 샘플된 토큰을 쓰지 않으므로 결과에 영향이 없다. 로그상 엔진 `seed=0`은 vLLM 기본값 |
| `stop` | 지정 안 함 | 한 토큰만 생성한다 |

`serve_logprobs.py`에는 `--max-tokens > 1`일 때 greedy 디코딩으로 텍스트까지 받는
자유 생성 경로도 있지만, 결과로 남아 있는 run은 모두 `max_tokens=1`이다.

## Chat template 처리

로컬 `prepare`에서 `tokenizer.apply_chat_template(chat, tokenize=True,
add_generation_prompt=True, enable_thinking=False)`로 user 한 턴만 넣어 토큰 id를 만든다.
system 프롬프트는 따로 넣지 않으며, 모델 템플릿이 기본으로 넣는 것은 그대로 둔다.

- `enable_thinking=False`: Qwen3 계열이 추론 블록으로 시작하면 답 문자가 첫 토큰
  자리에 오지 않는다. Qwen3 템플릿은 이 플래그를 보면 빈 `<think></think>` 블록을
  스스로 닫는다. 플래그를 읽지 않는 템플릿은 무시한다.
- gpt-oss: 템플릿이 이 플래그를 읽지 않고 `<|start|>assistant`에서 끝나서, 다음 토큰이
  채널 헤더가 되고 A/B/C가 상위 logprob 밖으로 밀려났다. 그래서 모델 이름 접두사로
  `<|channel|>final<|message|>`를 프롬프트 끝에 붙인다(`pipeline.py`의
  `ASSISTANT_PREFILL`, 커밋 4fd7376).
- 라벨 토큰: `tokenizer_checker.py`가 `"A"`와 `" A"` 중 단일 토큰으로 인코딩되는 쪽을
  고른다. 24개 모델 모두 공백 없는 `"A"` 쪽이 선택되었다(`meta.json`의
  `label_token_variant`).
- 실제로 모델에 들어간 프롬프트 한 개를 디코딩한 문자열이 이후 run의 `meta.json`
  `sample_prompt`에 남아 있다.

## 실행 기록으로 확인되는 것

엔진 로그가 레포에 남아 있는 것은 2026-07-25 래더의 **원 조건 Qwen2.5 0.5B, 1.5B, 3B**
세 run뿐이다(`results/serving_logs/ladder_20260725_195436/`). 로그의 `non-default args`에
`gpu_memory_utilization`, `tensor_parallel_size`, `max_num_seqs`가 없다는 것은 셋 다
vLLM 기본값(0.9, 1, 기본값)으로 돌았다는 뜻이고, 이는 yaml 값과 일치한다.

| 항목 | Qwen2.5-0.5B | Qwen2.5-1.5B | Qwen2.5-3B |
|---|---|---|---|
| vLLM | 0.10.2 (V1 엔진) | 0.10.2 | 0.10.2 |
| non-default args | dtype, max_model_len 1024, max_logprobs 50, revision | 같음 | 같음 |
| tensor_parallel_size | 1 | 1 | 1 |
| quantization | None | None | None |
| 모델 로딩 메모리 | 0.93 GiB | 2.89 GiB | 5.79 GiB |
| KV cache로 쓸 수 있는 메모리 | 18.89 GiB | 16.92 GiB | 14.01 GiB |
| GPU KV cache 크기 | 1,650,832 토큰 | 633,712 토큰 | 408,048 토큰 |
| 1,024 토큰 요청 기준 최대 동시성(vLLM 계산값) | 1612.14x | 618.86x | 398.48x |
| 엔진 초기화 시간 | 27.94초 | 32.50초 | 39.76초 |
| 기타 엔진 상태 | prefix caching 켜짐, chunked prefill(max_num_batched_tokens 8192), CUDA graph 사용(enforce_eager False), kv_cache_dtype auto | 같음 | 같음 |
| 프롬프트 수 | 31,372 | 31,372 | 31,372 |

로그에는 vLLM 진행 막대가 묶음마다 출력한 처리 속도 추정치도 찍혀 있지만, 이는 묶음
단위의 순간 추정치라 전체 처리량이나 지연 시간 측정으로 쓸 수 없다. 이 레포에는
throughput과 latency를 따로 잰 기록이 없다.

## 기록 없음

| 항목 | 상태 |
|---|---|
| GPU 모델과 장수 | 모든 run에서 기록 없음 |
| 위 3개 외 run(원 조건 21개, 강제 선택 조건 24개 전부)의 실제 엔진 인자 | 기록 없음. yaml 값은 스크립트 기본값이며 env로 override했는지 알 수 없다 |
| 위 3개 외 run의 vLLM 버전 | 기록 없음. Qwen3.5, gemma-4 같은 이후 모델은 다른 버전을 썼을 수 있으나 확인되지 않는다 |
| Qwen2.5-14B 등 큰 모델의 `TP`, `MAX_NUM_SEQS` | 기록 없음 |
| Qwen3.5-9B 강제 선택 조건 재실행 설정 | 첫 실행은 엔진 초기화에서 실패(GDN/Mamba 캐시 블록 부족, 커밋 21b45f9)하고 재실행되었으나 바꾼 설정은 기록 없음 |
| 체크포인트 자체의 가중치 형식을 엔진이 어떻게 적용했는지 | quantization 인자는 넘기지 않았고, 로그가 없는 모델은 엔진이 실제로 적용한 형식이 기록 없음 |
| throughput, latency, GPU 메모리 사용량 측정 | 기록 없음 |
