# vLLM 서빙 설정

이 문서는 BBQ 평가에서 모델을 vLLM으로 어떻게 띄웠는지, 어떤 값이 어디에 있는지,
그중 무엇이 실행 기록으로 확인되고 무엇이 확인되지 않는지를 정리한다.

## 하드웨어

연구실 GPU 서버의 NVIDIA GeForce RTX 3090 24GB(24,576 MiB) 2장, 드라이버 555.42.06이다(서버의
`nvidia-smi`, 2026-11-15 확인). 엔진 로그의 메모리 수치(아래 표, `gpu_memory_utilization`
0.9에서 모델과 KV cache를 합쳐 약 20 GiB)도 장당 24GB와 맞는다. run마다 한 장(TP 1) 또는 두
장(TP 2)을 썼다.

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

모델마다 `configs/serving/<model>.yaml`이 하나씩 있다(24개). 파일 안에 `original`과
`forced_choice` 두 블록이 있고, 각 블록에 그 조건의 run이 실제로 쓴 엔진 인자, runner 인자,
`CUDA_VISIBLE_DEVICES`, 근거(`provenance`: vLLM 버전, 서버의 로그 경로, 메모)를 적었다.
같은 모델이라도 조건마다 인자가 다른 경우가 많아서 블록을 나눴다. `scripts/serve.py`는
run 디렉터리의 `meta.json`(`"arm": "binary"` 여부)으로 블록을 고르고, 그 값을
`serve_logprobs.py` 인자로 바꿔 실행한다.

```bash
# 준비된 run 디렉터리를 입력으로 (조건과 revision은 그 run의 meta.json에서 읽는다)
python scripts/serve.py --config configs/serving/qwen2.5-3b-instruct.yaml \
    --run-dir results/runs/Qwen2.5-3B-Instruct_bfloat16_6db43b09

# 서버에 requests.jsonl만 있을 때는 조건을 직접 지정
python scripts/serve.py --config qwen2.5-3b-instruct.yaml --condition forced_choice \
    --input requests.jsonl --output responses.jsonl

# 실제 실행 없이 명령과 환경 변수만 출력
python scripts/serve.py --config ... --run-dir ... --dry-run
```

## 값의 근거

모든 값은 연구실 서버에 남은 vLLM 엔진 로그에서 읽었다. 각 run의 `*.2_serve.log`에 vLLM이
찍는 `non-default args`(기본값과 다른 인자만 나온다), 엔진 버전, `quantization`,
`tensor_parallel_size` 줄을 사용자가 서버에서 화면으로 확인해 가져왔다(2026-11-15). 서버
로그 파일은 크기가 커서 레포에 넣지 않았다. yaml의 `provenance.log`에 서버 쪽 경로가 있고,
2026-07-25 Qwen2.5 래더 로그만 `results/serving_logs/`에 커밋되어 있다.

예외는 원 조건 Qwen2.5-7B 하나다. 래더 스크립트가 생기기 전에 손으로 돌려서 엔진 로그가
남지 않았고, 값은 서버의 셸 히스토리(`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
python serve_logprobs.py ... --max-model-len 1024`)와 그 시점 `serve_logprobs.py`의 기본값에서
왔다. batch는 기본값 256이었고, 재개한 일부 실행에서는 64였다. batch는 결과에 영향이 없다.
이 run의 vLLM 버전은 기록이 없다.

## 엔진 인자 (`vllm.LLM(...)`)

| 인자 | 값 |
|---|---|
| `model`, `revision` | HF 모델 이름, 고정 commit sha (`prepare`가 `main`을 sha로 바꿔 `meta.json`에 기록) |
| `dtype` | `bfloat16` (48개 run 전부) |
| `max_model_len` | 1024 (48개 run 전부). 길이가 기록된 run 중 가장 긴 프롬프트는 279 토큰(gpt-oss-20b, 원 조건) |
| `gpu_memory_utilization` | 0.90. 원 조건 Ministral 3종만 0.80 |
| `tensor_parallel_size` | 1 또는 2 (아래 표) |
| `max_num_seqs` | 지정하지 않으면 vLLM 기본값 256(두 버전 모두 오류 메시지에서 확인). 일부 run은 64 |
| `max_logprobs` | 50 |
| `limit_mm_per_prompt` | `{"image": 0}`: 원 조건 Ministral, gemma-4, gpt-oss와 강제 선택 조건 24개 전부. 강제 선택 래더는 모든 모델에 `LIMIT_IMAGES=0`을 넘겼고, 텍스트 전용 모델에는 영향이 없다 |
| `quantization` | 인자로 넘긴 적은 없다. 엔진은 gpt-oss-20b에서만 체크포인트 설정에 따라 `gpt_oss_mxfp4`를 적용했고, 나머지 run은 로그에 `quantization=None` |

### run별 값

"원"은 원 조건, "강제"는 강제 선택 조건이다. 기타 열의 `img 0`은 `limit_mm_per_prompt={"image": 0}`이다.

| 모델 | 원: vLLM | TP | util | max_num_seqs | 기타 | 강제: vLLM | TP | util | max_num_seqs | 기타 |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b | 0.26.0 | 2 | 0.90 | 64 | img 0, gpt_oss_mxfp4 | 0.26.0 | 2 | 0.90 | 64 | img 0, gpt_oss_mxfp4 |
| gemma-4-E2B-it | 0.26.0 | 2 | 0.90 | 64 | img 0 | 0.26.0 | 1 | 0.90 | 64 | img 0 |
| gemma-4-E4B-it | 0.26.0 | 2 | 0.90 | 64 | img 0 | 0.26.0 | 1 | 0.90 | 64 | img 0 |
| gemma-4-12B-it | 0.26.0 | 2 | 0.90 | 64 | img 0 | 0.26.0 | 2 | 0.90 | 64 | img 0 |
| Llama-3.2-1B-Instruct | 0.10.2 | 1 | 0.90 | 기본 |  | 0.26.0 | 1 | 0.90 | 기본 | img 0 |
| Llama-3.2-3B-Instruct | 0.10.2 | 1 | 0.90 | 기본 |  | 0.26.0 | 1 | 0.90 | 기본 | img 0 |
| Llama-3.1-8B-Instruct | 0.10.2 | 1 | 0.90 | 기본 |  | 0.26.0 | 1 | 0.90 | 기본 | img 0 |
| Ministral-3-3B-Instruct-2512-BF16 | 0.26.0 | 2 | 0.80 | 64 | img 0 | 0.26.0 | 1 | 0.90 | 기본 | img 0 |
| Ministral-3-8B-Instruct-2512-BF16 | 0.26.0 | 2 | 0.80 | 64 | img 0 | 0.26.0 | 1 | 0.90 | 기본 | img 0 |
| Ministral-3-14B-Instruct-2512-BF16 | 0.26.0 | 2 | 0.80 | 64 | img 0 | 0.26.0 | 2 | 0.90 | 64 | img 0 |
| Qwen2.5-0.5B-Instruct | 0.10.2 | 1 | 0.90 | 기본 |  | 0.26.0 | 1 | 0.90 | 기본 | img 0 |
| Qwen2.5-1.5B-Instruct | 0.10.2 | 1 | 0.90 | 기본 |  | 0.26.0 | 1 | 0.90 | 기본 | img 0 |
| Qwen2.5-3B-Instruct | 0.10.2 | 1 | 0.90 | 기본 |  | 0.26.0 | 1 | 0.90 | 기본 | img 0 |
| Qwen2.5-7B-Instruct | 기록 없음 | 1 | 0.90 | 기본 | batch 256 | 0.26.0 | 1 | 0.90 | 기본 | img 0 |
| Qwen2.5-14B-Instruct | 0.10.2 | 2 | 0.90 | 64 |  | 0.26.0 | 2 | 0.90 | 64 | img 0 |
| Qwen3-0.6B | 0.10.2 | 2 | 0.90 | 기본 |  | 0.26.0 | 1 | 0.90 | 기본 | img 0 |
| Qwen3-1.7B | 0.10.2 | 2 | 0.90 | 기본 |  | 0.26.0 | 1 | 0.90 | 기본 | img 0 |
| Qwen3-4B | 0.10.2 | 2 | 0.90 | 기본 |  | 0.26.0 | 1 | 0.90 | 기본 | img 0 |
| Qwen3-8B | 0.10.2 | 2 | 0.90 | 기본 |  | 0.26.0 | 1 | 0.90 | 기본 | img 0 |
| Qwen3-14B | 0.26.0 | 2 | 0.90 | 64 |  | 0.26.0 | 2 | 0.90 | 64 | img 0 |
| Qwen3.5-0.8B | 0.26.0 | 2 | 0.90 | 64 |  | 0.26.0 | 1 | 0.90 | 기본 | img 0 |
| Qwen3.5-2B | 0.26.0 | 2 | 0.90 | 64 |  | 0.26.0 | 1 | 0.90 | 기본 | img 0 |
| Qwen3.5-4B | 0.26.0 | 2 | 0.90 | 64 |  | 0.26.0 | 1 | 0.90 | 기본 | img 0 |
| Qwen3.5-9B | 0.26.0 | 2 | 0.90 | 64 |  | 0.26.0 | 1 | 0.90 | 64 | img 0 |

TP 1인 run은 GPU 한 장(원 조건은 `GPU=1`, 강제 선택 조건은 `GPU=0`)을, TP 2인 run은 두
장(`GPU=0,1`)을 썼다. 같은 모델이 조건에 따라 TP가 다르거나(Ministral 3B와 8B, Qwen3 0.6B부터
8B, Qwen3.5, gemma-4 E2B와 E4B) vLLM 버전이 다르다(원 조건 Qwen2.5, Llama, Qwen3 0.6B부터
8B는 0.10.2, 강제 선택 조건은 모두 0.26.0). 두 조건을 비교할 때 이 차이가 logprob에 주는
영향은 확인하지 않았다.

### 설정이 바뀐 경위 (로그에 남은 실패 시도)

`batch_size`(기본 512)는 vLLM 인자가 아니라 `serve_logprobs.py`가 `llm.generate()` 한 번에
넘기는 프롬프트 수다. 실제로 동시에 스케줄되는 시퀀스 수는 vLLM 스케줄러의 `max_num_seqs`가
정한다.

- **Qwen2.5-14B (원 조건, 2026-07-28)**: TP 1, util 0.95로는 bf16 가중치(약 29GB)가 24GB 한
  장에 들어가지 않았다. TP 2로 나누자 이번에는 엔진 초기화 마지막의 sampler warmup(더미 요청
  256개, 요청마다 vocab 크기의 logits 행을 잡는다)에서 CUDA 메모리가 부족했다. 그래서
  `max_num_seqs`를 64로 낮춰 완료했다. `run_ladder.sh`의 `TP`, `MAX_NUM_SEQS` 옵션이 이때 생겼다(커밋 74b79f3, 64ddd15).
- **Ministral 3종 (원 조건, 2026-07-30)**: TP 2, util 0.9 시도가 여러 번 엔진 초기화에서
  실패했고, util 0.80, `max_num_seqs` 64로 완료했다.
- **Qwen3-14B, Qwen3.5 4종 (원 조건, 2026-07-31)**: 처음 래더(vLLM 0.10.2)에서 Qwen3-14B는
  메모리 부족으로 실패했고 Qwen3.5는 엔진 초기화까지 가지 못했다. vLLM 0.26.0 환경에서
  `max_num_seqs` 64로 다시 돌렸다. 앞서 끝난 Qwen3 0.6B부터 8B는 재실행에서 건너뛰었으므로
  0.10.2 결과가 남았다.
- **Qwen3.5-9B (강제 선택, 2026-08-18)**: `max_num_seqs (256) exceeds available Mamba cache
  blocks (89)`로 엔진 초기화에 실패해, `max_num_seqs` 64로 다시 돌렸다.

`max_tokens=1`이라 시퀀스가 한 스텝에 끝나므로 `max_num_seqs`를 낮춰도 손해가 작다는 것이
당시 판단이었다. 처리 시간을 비교해 확인하지는 않았다.

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
- gpt-oss 템플릿은 실행 날짜(`Current date: ...`)를 system 메시지에 넣으므로, 같은 설정으로 다시 돌려도 프롬프트가 그 한 줄만큼 달라진다.
- 라벨 토큰: `tokenizer_checker.py`가 `"A"`와 `" A"` 중 단일 토큰으로 인코딩되는 쪽을
  고른다. 24개 모델 모두 공백 없는 `"A"` 쪽이 선택되었다(`meta.json`의
  `label_token_variant`).
- 실제로 모델에 들어간 프롬프트 한 개를 디코딩한 문자열이 이후 run의 `meta.json`
  `sample_prompt`에 남아 있다.

## 엔진 상태 예시 (레포에 커밋된 로그)

레포에 전체 로그가 커밋된 것은 2026-07-25 래더의 원 조건 Qwen2.5 0.5B, 1.5B, 3B 세 run이다
(`results/serving_logs/ladder_20260725_195436/`). 이 로그에서는 인자 외의 엔진 상태도 볼 수 있다.

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
단위의 순간 추정치라 전체 처리량이나 지연 시간 측정으로 쓸 수 없다. throughput과 latency는
측정하지 않았다.

## 기록 없음, 측정하지 않음

| 항목 | 상태 |
|---|---|
| 원 조건 Qwen2.5-7B의 vLLM 버전과 엔진 상태 | 기록 없음 (래더 밖에서 수동 실행) |
| 레포에 커밋되지 않은 서버 로그의 엔진 상태(KV cache 크기, 초기화 시간 등) | 서버에만 있다. 이 문서에는 인자와 버전만 옮겼다 |
| vLLM 0.26.0이 깔린 환경 | 서버의 conda env `bbq`에는 현재 vLLM 0.10.2(transformers 4.57.6, torch 2.8.0+cu128)가 있다. 0.26.0으로 돌린 run이 어느 환경을 썼는지는 기록 없음 |
| 조건 사이 vLLM 버전, TP 차이가 logprob에 주는 영향 | 확인하지 않았다 |
| throughput, latency, GPU 메모리 사용량 | 측정하지 않았다 |
