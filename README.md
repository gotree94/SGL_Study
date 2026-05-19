# SGLang 로컬 설치 및 운영 종합 가이드

> **SGLang**은 대규모 언어 모델(LLM) 및 멀티모달 모델을 위한 고성능 서빙 프레임워크입니다.
> 최신 버전: **v0.5.11** (2026-05-05 기준)

---

## 목차

1. [SGLang 개요 및 아키텍처](#1-sglang-개요-및-아키텍처)
2. [하드웨어 요구사항](#2-하드웨어-요구사항)
3. [운영체제 선택](#3-운영체제-선택)
4. [사전 설치 준비](#4-사전-설치-준비)
5. [설치 방법](#5-설치-방법)
6. [모델 서빙 및 실행](#6-모델-서빙-및-실행)
7. [고급 설정: 병렬처리 & 분산](#7-고급-설정-병렬처리--분산)
8. [Evaluation Harness (sgl-eval)](#8-evaluation-harness-sgl-eval)
9. [오케스트레이션: Docker & Kubernetes](#9-오케스트레이션-docker--kubernetes)
10. [운영 팁 & 트러블슈팅](#10-운영-팁--트러블슈팅)
11. [참고 자료](#11-참고-자료)

---

## 1. SGLang 개요 및 아키텍처

### 1.1 SGLang이란?

SGLang은 LLM/멀티모달 모델의 **고속 추론 서빙**을 위한 프레임워크입니다. 기존 vLLM, TensorRT-LLM과 비교해 다음과 같은 장점이 있습니다:

- **RadixAttention**: Prefix caching을 통한 KV 캐시 재사용으로 TTFT(Time-To-First-Token) 최적화
- **Chunked Prefill**: 큰 프리필 요청을 작은 청크로 분할하여 효율적 처리
- **Continuous Batching**: 실시간 요청 도착에 따른 동적 배칭
- **CUDA Graph 최적화**: 반복 연산을 그래프로 캡처해 오버헤드 제거
- **Tensor/Data/Expert Parallelism**: 다중 GPU 분산 지원
- **OpenAI 호환 API**: 기존 코드 수정 없이 드롭인 교체 가능

### 1.2 시스템 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                 Frontend Language (SGLang)           │
│    - Structured generation primitives                │
│    - Control flow (fork, join)                       │
│    - Constrained decoding (regex, JSON schema)       │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│              HTTP / gRPC API Server                  │
│    - OpenAI 호환 엔드포인트 (/v1/chat/completions)    │
│    - Request routing & authentication                │
│    - Native SGLang API (/generate)                   │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│          Runtime (SRT - SGLang Runtime)              │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Scheduler │  │  Memory  │  │  Model Runner    │  │
│  │ · 동적배칭 │  │  Manager │  │ · Forward Pass   │  │
│  │ · 연속배칭 │  │ · KV Pool│  │ · Sampling       │  │
│  │ · Prefix  │  │ · Radix  │  │ · CUDA Graph     │  │
│  │  Caching  │  │   Tree   │  │                  │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 1.3 주요 컴포넌트

| 컴포넌트 | 역할 | 설명 |
|---------|------|------|
| **Engine** | 진입점 | 모델 초기화, 요청 라이프사이클 관리, IPC 조정 |
| **Scheduler** | 배칭 엔진 | 동적/연속 배칭, Prefix caching, Chunked prefill |
| **Memory Manager** | 메모리 관리 | Token-to-KV pool, 사전 할당 GPU 메모리, Radix tree |
| **Model Runner** | 모델 실행 | 가중치 로드, Forward pass (prefill + decode), Sampling |
| **Attention Backend** | Attention 최적화 | FlashInfer (기본), FlashAttention, Triton |

---

## 2. 하드웨어 요구사항

### 2.1 최소 사양

| 구성 요소 | 최소 사양 | 권장 사양 |
|---------|----------|----------|
| **GPU** | NVIDIA GPU (Compute Capability ≥ 7.0) | NVIDIA A100(80GB) / H100 / RTX 4090 |
| **GPU 메모리** | 16GB (7B 모델 FP16 기준) | 80GB (70B 모델) / 24GB (13B 모델 Q4) |
| **CPU** | 8코어 이상 | 32코어 이상 (AMD EPYC / Intel Xeon) |
| **RAM** | 32GB | 128GB 이상 |
| **스토리지** | SSD 50GB (모델 제외) | NVMe SSD 500GB+ (모델 저장) |
| **네트워크** | - | InfiniBand / 100GbE (멀티노드시) |
| **CUDA** | CUDA 12.1+ | CUDA 12.4+ / 13.0 |

> **Compute Capability 7.0+** = Tesla V100, RTX 20xx, GTX 16xx 이상
> - sm75 (Turing): T4, RTX 2060/2070/2080
> - sm80 (Ampere): A100, A30
> - sm86 (Ampere): RTX 3060/3070/3080/3090
> - sm89 (Ada Lovelace): RTX 4060/4070/4080/4090, L40S
> - sm90 (Hopper): H100, H200
> - sm100 (Blackwell): B200, B100
> - sm103 (Blackwell Ultra): B300, GB300

### 2.2 모델 크기별 필요 VRAM (추정치)

| 모델 크기 | FP16 | INT8 (AWQ/GPTQ) | INT4 (AWQ/GPTQ) |
|----------|------|-----------------|-----------------|
| 1.5B | ~4GB | ~2.5GB | ~2GB |
| 7B | ~16GB | ~9GB | ~5.5GB |
| 8B | ~18GB | ~10GB | ~6GB |
| 13B | ~28GB | ~15GB | ~8GB |
| 22B | ~45GB | ~24GB | ~13GB |
| 34B | ~68GB | ~36GB | ~19GB |
| 70B | ~140GB | ~75GB | ~40GB |
| 120B | ~240GB | ~128GB | ~68GB |
| DeepSeek-V3 (671B MoE) | ~404GB (BF16) | ~202GB (FP8) | - |

> **참고**: 위 수치는 모델 가중치만의 메모리이며, KV 캐시, activation, CUDA 그래프 등을 위한 추가 메모리가 필요합니다. 일반적으로 모델 크기의 **1.2~1.5배**의 VRAM을 확보하는 것이 좋습니다.

### 2.3 GPU 추천 표

| 사용 사례 | 추천 GPU | VRAM | 예상 처리량 |
|----------|---------|------|-----------|
| 개인/소규모 (7-13B) | RTX 4090 | 24GB | ~2000 tok/s (7B) |
| 중규모 (13-34B) | A100-40G / L40S | 48GB | ~1500 tok/s (13B) |
| 대규모 (70B) | A100-80G × 2 (TP=2) | 160GB | ~800 tok/s (70B) |
| 초대규모 (70B+) | H100-80G × 4 (TP=4) | 320GB | ~2000 tok/s (70B) |
| MoE 모델 (DeepSeek) | H100-80G × 8 | 640GB | 모델 의존적 |

---

## 3. 운영체제 선택

### 3.1 OS별 지원 현황

| OS | 지원 수준 | 비고 |
|---|---------|------|
| **Ubuntu 22.04 / 24.04 LTS** | ✅ **최적** | 공식 지원, NVIDIA CUDA 최상의 호환성 |
| **Debian 12** | ✅ 권장 | 안정성 우수 |
| **RHEL 9 / Rocky 9** | ✅ 지원 | 엔터프라이즈 환경 |
| **Windows 11 (WSL2)** | ⚠️ 부분 지원 | WSL2 + Ubuntu로 사용 가능 |
| **macOS (Apple Silicon)** | ⚠️ 실험적 (MLX) | `SGLANG_USE_MLX=1` 필요, 성능 제한적 |
| **Windows (Native)** | ❌ 미지원 | CUDA 의존성 문제, Docker/WSL2 권장 |

### 3.2 OS 선택 가이드

#### 🏆 최우선: Ubuntu 24.04 LTS
```
이유:
- NVIDIA CUDA 12.x / 13.0 완벽 지원
- FlashInfer, sgl-kernel 등 네이티브 컴파일 지원
- Docker, Kubernetes 완벽 호환
- 커뮤니티/공식 문서의 모든 예제가 Ubuntu 기준
```

#### Windows 사용자: WSL2 + Ubuntu
```
Windows 11에서 SGLang을 사용하려면 WSL2(Windows Subsystem for Linux 2)가 필수입니다.
CUDA는 Windows 호스트에 설치해도 WSL2 내에서 접근 가능합니다.
```

#### macOS 사용자: MLX 백엔드 (실험적)
```
Apple Silicon(M1/M2/M3/M4)에서 SGLANG_USE_MLX=1 환경변수로 사용 가능하나,
아직 실험 단계이며 NVIDIA GPU 대비 성능이 낮습니다.
추론 자체보다는 개발/테스트 목적으로 권장합니다.
```

---

## 4. 사전 설치 준비

### 4.1 NVIDIA 드라이버 및 CUDA 설치

#### Ubuntu 24.04 예시

```bash
# 1. NVIDIA 드라이버 설치
sudo apt update
sudo apt install -y nvidia-driver-550  # 또는 최신 버전
sudo reboot

# 2. 설치 확인
nvidia-smi
# 출력 예:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 550.90.07    Driver Version: 550.90.07    CUDA Version: 12.4     |
# +-----------------------------------------------------------------------------+

# 3. CUDA Toolkit 설치 (SGLang v0.5.11부터 CUDA 13 기본)
# CUDA 12.4 권장 (안정적)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4

# 4. CUDA 환경변수 설정
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.4' >> ~/.bashrc
source ~/.bashrc

# 5. 확인
nvcc --version
```

#### Windows + WSL2 설정

```powershell
# Windows 호스트에서
# 1. 최신 NVIDIA Game Ready 드라이버 설치 (WSL2 CUDA 지원)
#    https://www.nvidia.com/download/index.aspx

# 2. WSL2 설치
wsl --install -d Ubuntu-24.04

# 3. WSL2 내부에서 CUDA 확인
wsl
nvidia-smi  # GPU 정보가 보여야 함
```

### 4.2 Python 환경 설정

```bash
# Python 3.10+ 필요 (3.11 또는 3.12 권장)
python3 --version

# 가상환경 생성 (권장)
python3 -m venv ~/sglang-env
source ~/sglang-env/bin/activate

# pip 및 uv 설치 (uv가 더 빠름)
pip install --upgrade pip
pip install uv
```

### 4.3 GPU 호환성 사전 체크 스크립트

설치 전 다음 스크립트로 환경을 검증하세요:

```python
# check_gpu_env.py
import subprocess
import sys

def check_nvidia_smi():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ nvidia-smi: 정상 동작")
            # GPU 정보 파싱
            for line in result.stdout.split('\n'):
                if 'GeForce' in line or 'Tesla' in line or 'A100' in line or 'H100' in line:
                    print(f"   감지된 GPU: {line.strip()}")
            return True
        else:
            print("❌ nvidia-smi: 실패")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi: NVIDIA 드라이버 미설치")
        return False
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi: 시간 초과")
        return False

def check_cuda():
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if "release" in line:
                    print(f"✅ CUDA Toolkit: {line.strip()}")
                    return True
        print("❌ nvcc: CUDA Toolkit 미설치")
        return False
    except FileNotFoundError:
        print("❌ nvcc: CUDA Toolkit 미설치 또는 PATH 미설정")
        return False

def check_torch_cuda():
    try:
        import torch
        print(f"✅ PyTorch 버전: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
            print(f"   Compute Capability: {torch.cuda.get_device_capability(0)}")
            print(f"   CUDA 버전 (PyTorch): {torch.version.cuda}")
            cc = torch.cuda.get_device_capability(0)
            if cc[0] < 7:
                print("❌ Compute Capability < 7.0: SGLang 미지원")
                return False
            print("✅ Compute Capability 7.0+ : SGLang 지원")
            return True
        else:
            print("❌ CUDA 사용 불가")
            return False
    except ImportError:
        print("❌ PyTorch 미설치")
        return False
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("SGLang GPU 환경 체크")
    print("=" * 50)
    
    gpu_ok = check_nvidia_smi()
    cuda_ok = check_cuda()
    
    print()
    print("-" * 50)
    print("PyTorch CUDA 체크")
    print("-" * 50)
    torch_ok = check_torch_cuda()
    
    print()
    print("=" * 50)
    if gpu_ok and torch_ok:
        print("✅ SGLang 설치 준비 완료!")
    else:
        print("❌ 환경이 SGLang 요구사항을 충족하지 않습니다.")
        print("   위 에러를 확인하고 수정하세요.")
    print("=" * 50)
```

실행:
```bash
python check_gpu_env.py
```

---

## 5. 설치 방법

### 5.1 pip/uv로 설치 (권장)

#### 기본 설치 (CUDA 12.x)

```bash
# 방법 A: uv 사용 (권장, 더 빠름)
source ~/sglang-env/bin/activate
pip install --upgrade pip
pip install uv
uv pip install sglang

# 방법 B: pip 사용
pip install --upgrade pip
pip install sglang

# 설치 확인
python -c "import sglang; print(sglang.__version__)"
```

#### CUDA 13 환경 설치

```bash
# CUDA 13 환경에서는 Docker 권장
# Docker 미사용시:
uv pip install sglang

# sglang-kernel wheel 설치 (필요시)
# https://github.com/sgl-project/whl/releases 에서 버전 확인
```

### 5.2 소스에서 설치 (개발용)

```bash
# 최신 릴리스 브랜치 클론
git clone -b v0.5.11 https://github.com/sgl-project/sglang.git
cd sglang

# 설치
pip install --upgrade pip
pip install -e "python"

# 개발 컨테이너 사용시
# docker pull lmsysorg/sglang:dev
```

### 5.3 Docker로 설치 (프로덕션 권장)

```bash
# 최신 안정 버전
docker pull lmsysorg/sglang:latest

# 특정 CUDA 버전 이미지
docker pull lmsysorg/sglang:latest-cu129  # CUDA 12.9
docker pull lmsysorg/sglang:latest-cu130  # CUDA 13.0

# 런타임 전용 (경량, 프로덕션)
docker pull lmsysorg/sglang:latest-runtime  # 약 40% 작음
```

### 5.4 설치 검증

```bash
# 기본 서빙 테스트 (작은 모델로)
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 30000

# 별도 터미널에서 요청 테스트
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0
    }'
```

---

## 6. 모델 서빙 및 실행

### 6.1 기본 서버 실행

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.8
```

### 6.2 주요 서버 인자

#### 모델 관련

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--model-path` | HF 모델 ID 또는 로컬 경로 | 필수 |
| `--load-format` | 가중치 포맷 (auto/pt/safetensors/gguf) | auto |
| `--trust-remote-code` | HF 원격 코드 실행 허용 | False |
| `--tokenizer-path` | 토크나이저 경로 (별도 지정시) | model-path와 동일 |
| `--context-length` | 최대 컨텍스트 길이 | 모델 기본값 |
| `--quantization` | 양자화 (awq/gptq/fp8/int8) | None |

#### 성능 관련

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--tp` | Tensor Parallelism 크기 | 1 |
| `--dp` | Data Parallelism 크기 | 1 |
| `--nnodes` | 노드 수 | 1 |
| `--mem-fraction-static` | GPU 메모리 사용률 (0~1) | 0.9 |
| `--max-running-requests` | 최대 동시 요청 수 | 200 |
| `--max-total-tokens` | 최대 총 토큰 수 | 자동 |
| `--schedule-policy` | 스케줄링 정책 (lpm/lof/random) | lpm |
| `--attention-backend` | Attention 백엔드 (flashinfer/triton/flash-attention) | flashinfer |
| `--sampling-backend` | Sampling 백엔드 (pytorch/cuda/graph) | cuda |

#### 네트워크/기타

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--host` | 바인딩 주소 | 127.0.0.1 |
| `--port` | 포트 번호 | 30000 |
| `--api-key` | API 키 인증 | None |
| `--log-level` | 로그 레벨 | info |

### 6.3 Inference 사용 예제

#### OpenAI 호환 API (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY"
)

# Chat Completion
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    temperature=0.7,
    max_tokens=512
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Write a poem about AI."}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

#### Offline Engine API

```python
import sglang as sgl

# 모델 로드
llm = sgl.Engine(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    mem_fraction_static=0.8,
)

# 추론
response = llm.generate(
    input_text="What is the capital of France?",
    max_new_tokens=128,
    temperature=0,
)
print(response)

# 배치 추론
responses = llm.generate(
    input_text=[
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the capital of Brazil?",
    ],
    max_new_tokens=64,
)
for r in responses:
    print(r)

# 정리
llm.shutdown()
```

### 6.4 양자화 사용

```bash
# AWQ 양자화 모델
python3 -m sglang.launch_server \
    --model-path TheBloke/Llama-2-7B-Chat-AWQ \
    --quantization awq \
    --host 0.0.0.0 \
    --port 30000

# FP8 (H100/H200 지원)
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --quantization fp8 \
    --host 0.0.0.0 \
    --port 30000
```

---

## 7. 고급 설정: 병렬처리 & 분산

### 7.1 Tensor Parallelism (단일 노드, 다중 GPU)

```bash
# 2개 GPU 사용
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 2 \
    --host 0.0.0.0 --port 30000

# 4개 GPU 사용
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 4 \
    --host 0.0.0.0 --port 30000
```

### 7.2 Data Parallelism (DP)

```bash
# 2개 GPU 데이터 병렬 처리
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --dp 2 \
    --host 0.0.0.0 --port 30000
```

### 7.3 Tensor + Data 병렬 혼합

```bash
# 총 4GPU: TP=2 × DP=2
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 2 --dp 2 \
    --host 0.0.0.0 --port 30000
```

### 7.4 멀티노드 분산 (Tensor Parallelism)

```bash
# 노드 1 (sgl-dev-0)
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 4 \
    --nnodes 2 \
    --node-rank 0 \
    --dist-init-addr sgl-dev-0:25000 \
    --host 0.0.0.0 --port 30000

# 노드 2 (sgl-dev-1)
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 4 \
    --nnodes 2 \
    --node-rank 1 \
    --dist-init-addr sgl-dev-0:25000 \
    --host 0.0.0.0 --port 30001
```

### 7.5 Prefill-Decode Disaggregation (분리형 서빙)

Prefill(프리필)과 Decode(디코드)를 별도 GPU로 분리하여 처리량 최적화:

```bash
# Prefill 노드
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --node-prefill \
    --tp 4 \
    --host 0.0.0.0 --port 30000

# Decode 노드
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --node-decode \
    --tp 4 \
    --host 0.0.0.0 --port 30001
```

---

## 8. Evaluation Harness (sgl-eval)

### 8.1 sgl-eval 개요

**sgl-eval**은 SGLang 공식 평가 도구로, OpenAI 호환 엔드포인트에 연결하여 모델 성능을 측정합니다.

```
┌──────────────────────────────────────────────┐
│  sgl-eval                                     │
│  CLI / Sampler / Metrics                      │
│  evals/ (벤치마크 설정)                        │
├──────────────────────────────────────────────┤
│  NeMo-Skills (vendored)                       │
│  Math Grader / Evaluator / Dataset prompts    │
└──────────────────────────────────────────────┘
```

### 8.2 설치

```bash
pip install sgl-eval
```

### 8.3 사용법

```bash
# 지원 벤치마크 목록
sgl-eval list
sgl-eval list -v  # 상세 보기

# 엔드포인트 연결 확인
sgl-eval ping --endpoint http://localhost:30000/v1

# 평가 실행
sgl-eval run gsm8k \
    --endpoint http://localhost:30000/v1 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --n-repeats 1 \
    --out-dir ./eval_results

# 여러 벤치마크
sgl-eval run mmlu \
    --endpoint http://localhost:30000/v1 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --out-dir ./eval_results
```

### 8.4 Preset 사용

```yaml
# ~/.sgl_eval/presets/my-llama.yaml
benchmark: gsm8k
endpoint: http://localhost:30000/v1
model: meta-llama/Llama-3.1-8B-Instruct
n_repeats: 3
sampling:
  temperature: 0.0
  max_tokens: 1024
```

```bash
sgl-eval run --preset my-llama
```

### 8.5 지원 벤치마크

| 벤치마크 | 설명 |
|---------|------|
| gsm8k | 수학 추론 (초등학교 수준) |
| math500 | 고급 수학 |
| mmlu | 57개 과목 지식 |
| mmlu_pro | MMLU 확장 |
| gpqa | 박사 수준 과학 |
| humaneval | 코드 생성 |
| ifeval | 명령어 수행 |
| longbench | 긴 컨텍스트 이해 |

---

## 9. 오케스트레이션: Docker & Kubernetes

### 9.1 Docker Compose

```yaml
# docker-compose.yml
services:
  sglang:
    image: lmsysorg/sglang:latest-runtime
    container_name: sglang-server
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
    restart: always
    network_mode: host  # RDMA 필요시
    environment:
      - HF_TOKEN=${HF_TOKEN:-}
    entrypoint: python3 -m sglang.launch_server
    command: >
      --model-path meta-llama/Llama-3.1-8B-Instruct
      --host 0.0.0.0
      --port 30000
      --mem-fraction-static 0.85
    ulimits:
      memlock: -1
      stack: 67108864
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:30000/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
# 실행
export HF_TOKEN=hf_xxxxxxxx
docker compose up -d

# 로그 확인
docker compose logs -f
```

### 9.2 Kubernetes (LWS - LeaderWorkerSet)

SGLang 공식 Kubernetes 배포는 **LeaderWorkerSet (LWS)** 커스텀 리소스를 사용합니다.

```yaml
# sglang-k8s.yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: sglang-llama
spec:
  replicas: 1
  leaderWorkerTemplate:
    size: 2  # 워커 수
    restartPolicy: RecreateGroupOnPodRestart
    leaderTemplate:
      metadata:
        labels:
          role: leader
      spec:
        dnsPolicy: ClusterFirstWithHostNet
        hostNetwork: true
        hostIPC: true
        containers:
          - name: sglang-leader
            image: lmsysorg/sglang:latest-runtime
            securityContext:
              privileged: true
            env:
              - name: NCCL_IB_GID_INDEX
                value: "3"
              - name: HF_TOKEN
                valueFrom:
                  secretKeyRef:
                    name: hf-token
                    key: token
            command:
              - python3
              - -m
              - sglang.launch_server
              - --model-path
              - meta-llama/Llama-3.1-70B-Instruct
              - --mem-fraction-static
              - "0.90"
              - --max-running-requests
              - "100"
              - --tp
              - "4"
              - --dist-init-addr
              - $(LWS_LEADER_ADDRESS):20000
              - --nnodes
              - $(LWS_GROUP_SIZE)
              - --node-rank
              - $(LWS_WORKER_INDEX)
              - --trust-remote-code
              - --host
              - "0.0.0.0"
              - --port
              - "30000"
            resources:
              limits:
                nvidia.com/gpu: "4"
            ports:
              - containerPort: 30000
            readinessProbe:
              tcpSocket:
                port: 30000
              initialDelaySeconds: 30
              periodSeconds: 10
            volumeMounts:
              - mountPath: /dev/shm
                name: dshm
              - mountPath: /root/.cache/huggingface
                name: model-cache
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
          - name: model-cache
            hostPath:
              path: /mnt/models
    workerTemplate:
      spec:
        dnsPolicy: ClusterFirstWithHostNet
        hostNetwork: true
        hostIPC: true
        containers:
          - name: sglang-worker
            image: lmsysorg/sglang:latest-runtime
            securityContext:
              privileged: true
            env:
              - name: NCCL_IB_GID_INDEX
                value: "3"
              - name: HF_TOKEN
                valueFrom:
                  secretKeyRef:
                    name: hf-token
                    key: token
            command:
              - python3
              - -m
              - sglang.launch_server
              - --model-path
              - meta-llama/Llama-3.1-70B-Instruct
              - --mem-fraction-static
              - "0.90"
              - --max-running-requests
              - "100"
              - --tp
              - "4"
              - --dist-init-addr
              - $(LWS_LEADER_ADDRESS):20000
              - --nnodes
              - $(LWS_GROUP_SIZE)
              - --node-rank
              - $(LWS_WORKER_INDEX)
              - --trust-remote-code
              - --host
              - "0.0.0.0"
              - --port
              - "30000"
            resources:
              limits:
                nvidia.com/gpu: "4"
            volumeMounts:
              - mountPath: /dev/shm
                name: dshm
              - mountPath: /root/.cache/huggingface
                name: model-cache
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
          - name: model-cache
            hostPath:
              path: /mnt/models
```

```bash
# LWS 설치 (최초 1회)
kubectl apply --server-side -f https://github.com/kubernetes-sigs/lws/releases/download/v0.4.0/manifest.yaml

# SGLang 배포
kubectl apply -f sglang-k8s.yaml

# 서비스 노출 (NodePort)
kubectl expose pod sglang-llama-0-leader \
    --name sglang-service \
    --port 30000 \
    --target-port 30000 \
    --type NodePort

# 상태 확인
kubectl get pods -l leaderworkerset=sglang-llama
```

### 9.3 SGLang Model Gateway (라우터)

대규모 배포를 위한 **Model Gateway**는 여러 모델 워커를 통합 관리하는 라우터입니다.

```bash
# 게이트웨이 실행 (워커와 함께)
python3 -m sglang.gateway \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --worker-type sglang \
    --worker-count 2 \
    --host 0.0.0.0 --port 30000

# gRPC 라우터 (고성능)
python3 -m sglang.gateway \
    --enable-igw \
    --tokenizer-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 30000
```

---

## 10. 운영 팁 & 트러블슈팅

### 10.1 성능 최적화

```bash
# 권장 설정 (A100/H100 기준)
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --mem-fraction-static 0.90 \
    --max-running-requests 200 \
    --schedule-policy lpm \
    --attention-backend flashinfer \
    --sampling-backend cuda

# FP8 (H100 전용, 대폭 속도 향상)
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --quantization fp8 \
    --mem-fraction-static 0.90
```

### 10.2 VRAM 부족시

```bash
# 메모리 사용률 낮춤
--mem-fraction-static 0.7  # 기본 0.9에서 감소

# 최대 컨텍스트 길이 제한
--context-length 4096       # 기본 32768에서 축소

# 최대 배치 크기 제한
--max-running-requests 50   # 기본 200에서 축소

# 양자화 사용
--quantization awq          # INT4 양자화로 VRAM 50%+ 절약
```

### 10.3 일반적인 문제와 해결

| 문제 | 증상 | 해결 |
|------|------|------|
| **CUDA Out of Memory** | `CUDA out of memory` | `--mem-fraction-static` 감소, 양자화 사용 |
| **FlashInfer 오류** | `FlashInfer 관련 에러` | `--attention-backend triton --sampling-backend pytorch` |
| **Peer Access 오류** | `peer access is not supported` | `--enable-p2p-check` 추가 |
| **CUDA_HOME 미설정** | `CUDA_HOME environment variable is not set` | `export CUDA_HOME=/usr/local/cuda-12.4` |
| **NCCL 데드락** | 멀티 GPU에서 멈춤 | `--disable-cuda-graph` 추가 |
| **CUDA 버전 불일치** | 드라이버 vs 툴킷 버전 미스매치 | nvidia-smi로 드라이버 버전 확인 후 CUDA 툴킷 재설치 |
| **모델 로딩 실패** | `CUDA error: device-side assert` | 모델과 토크나이저 호환성 확인, `--trust-remote-code` 추가 |

### 10.4 모니터링

```bash
# SGLang 서버 상태 확인
curl http://localhost:30000/health
curl http://localhost:30000/get_model_info
curl http://localhost:30000/server_info

# 실시간 모니터링 (별도 터미널)
watch -n 1 nvidia-smi

# 로그 수준 설정
# --log-level debug  # 상세 로그
# --log-level info   # 기본
# --log-level warn   # 경고 이상만
```

### 10.5 보안 권장사항

```bash
# API 키 설정
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --api-key "your-secret-api-key"

# Docker 프로덕션 (비특권 사용자)
# privileged: false (기본값)
# 필요한 최소 권한만 부여

# 방화벽 설정
# 30000번 포트만 허용
```

---

## 11. 전체 구성 예시: Step-by-Step

### 시나리오: Ubuntu 24.04 + RTX 4090 + Llama 3.1 8B

```bash
# ============================================
# 1단계: OS 및 드라이버 설치
# ============================================
# Ubuntu 24.04 LTS 설치
# NVIDIA 드라이버 550+ 설치

sudo apt update && sudo apt upgrade -y
sudo apt install -y nvidia-driver-550
sudo reboot

# ============================================
# 2단계: CUDA 및 필수 도구 설치
# ============================================
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4

# 환경변수 설정
cat >> ~/.bashrc << 'EOF'
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.4
EOF
source ~/.bashrc

# Docker 설치 (옵션)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

# ============================================
# 3단계: Python 및 SGLang 설치
# ============================================
sudo apt install -y python3 python3-pip python3-venv
python3 -m venv ~/sglang-env
source ~/sglang-env/bin/activate

pip install --upgrade pip
pip install uv
uv pip install sglang

# GPU 환경 확인
python check_gpu_env.py

# ============================================
# 4단계: SGLang 서버 실행
# ============================================
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.85 \
    --trust-remote-code

# ============================================
# 5단계: API 호출 테스트
# ============================================
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Hello! Who are you?"}],
        "temperature": 0.7,
        "max_tokens": 100
    }'

# ============================================
# 6단계: 평가 실행
# ============================================
pip install sgl-eval
sgl-eval run gsm8k \
    --endpoint http://localhost:30000/v1 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --n-repeats 1 \
    --out-dir ./eval_results

# ============================================
# 7단계: Docker Compose로 프로덕션 배포
# ============================================
cat > docker-compose.yml << 'EOF'
services:
  sglang:
    image: lmsysorg/sglang:latest-runtime
    container_name: sglang-llama
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
    restart: always
    network_mode: host
    entrypoint: python3 -m sglang.launch_server
    command: >
      --model-path meta-llama/Llama-3.1-8B-Instruct
      --host 0.0.0.0
      --port 30000
      --mem-fraction-static 0.85
      --trust-remote-code
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:30000/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
EOF

docker compose up -d
```

---

## 12. 참고 자료

| 자료 | 링크 |
|------|------|
| 공식 설치 가이드 | https://sgl-project.github.io/get_started/install.html |
| GitHub 저장소 | https://github.com/sgl-project/sglang |
| 서버 인자 문서 | https://docs.sglang.io/advanced_features/server_arguments.html |
| Docker 배포 가이드 | https://sgl-project-sglang-93.mintlify.app/deployment/docker |
| K8s 배포 가이드 | https://sgl-project.github.io/references/multi_node_deployment/deploy_on_k8s.html |
| sgl-eval 저장소 | https://github.com/sgl-project/sgl-eval |
| NVIDIA CUDA 다운로드 | https://developer.nvidia.com/cuda-downloads |
| HuggingFace 모델 | https://huggingface.co/models |

---

> **문서 작성일**: 2026-05-19
> **SGLang 최신 버전**: v0.5.11
> **문서 작성자**: Sisyphus
