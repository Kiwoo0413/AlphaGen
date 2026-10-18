# AlphaGen (알파젠)

**AlphaGen**은 비전-언어 모델(VLM) 기반의 객체 그라운딩, SAM2 고정밀 분할, LTX-2.5 비디오 디퓨전(IC-LoRA) 모델, 그리고 CompMatte 알고리즘 기반의 코어/엣지 분리 매트 정제를 통합한 차세대 제어형 알파 생성 시스템입니다.

Foundry Nuke 기즈모 환경에서 완벽하게 구동되며, 향후 웹 UI(FastAPI/Gradio) 및 학술 연구(논문)로 확장 가능한 이원화(Dual-Runtime) 아키텍처로 설계되었습니다.

---

## 🏗️ 시스템 아키텍처 (Pipeline Architecture)

```mermaid
flowchart TD
    A["첫 프레임 + 사용자 자연어 프롬프트"] --> B["Step 1-1. VLM 객체 그라운딩 (Gemini Flash)<br/>2D BBox 및 대상 텍스트 설명"]
    B --> C["Step 1-2. 로컬 SAM 2.1 (Hiera-Large)<br/>참조 프레임 1.2배 확장 픽셀 분할 마스크"]
    C --> D["Nuke 뷰어 사전 검증<br/>'Target Selection (VLM)' 뷰에서 즉시 확인"]
    D --> E["Step 2-1. 대상 조건 처리 (LTX Alpha Conditioning)<br/>공간 Attention Mask + Prompt Embedding"]
    E --> F["Step 2-2. LTX-2.5 IC-LoRA 인퍼런스<br/>49프레임 청크 & 17프레임 코사인 오버랩 무제한 연속 생성"]
    F --> G["Step 3. CompMatte CPU 정제 엔진<br/>밀리초 단위 Core Solidification + Edge Restoration"]
    G --> H["Step 4. 최종 합성용 알파 및 RGBA 출력<br/>Final Alpha / Raw LTX Alpha / Premultiplied RGBA"]
```

---

## 📁 프로젝트 파일 및 폴더 구조

정밀 검수를 거쳐 불필요한 임시 파일 및 프로브 스크립트를 모두 정리하고 다음과 같이 핵심 모듈 위주로 정돈되었습니다:

```text
AlphaGen/
├── gizmos/
│   └── AlphaGen.gizmo              # 리팩토링된 Nuke 기즈모 (4단계 선형 단일 탭, 단계별 구분선, Stop 버튼)
├── alphagen_nuke.py                # Nuke API & GUI 브릿지 (비동기 작업 제어, 프레임 입출력, 프로세스 트리 종료)
├── alphagen_vlm.py                 # VLM (Gemini Flash) 객체 그라운딩 & SAM2 분할 모듈
├── alphagen_worker.py              # 외부 Python 독립 AI 워커 (Nuke GIL 분리, 49프레임 청크/17프레임 오버랩)
├── alphagen_matte.py               # CPU 코어/엣지 분리 매트 정제 엔진 (순수 수학/OpenCV)
├── load_alphagen.py                # Nuke Script Editor 원클릭 실행 스크립트
├── init.py / menu.py               # Nuke 패키지 자동 경로 등록
├── experiment_config.json          # 모델 가중치 및 외부 Python 런타임 설정
├── NUKE_README.md                  # Nuke 아티스트용 상세 사용 매뉴얼
├── README.md                       # 프로젝트 종합 기술 문서 (본 파일)
│
├── tools/                          # 필수 실행 및 검증 도구
│   ├── run_experiment.py           # LTX 인퍼런스 실행 드라이버 (49프레임 청크 CLI 인자 자동 주입)
│   ├── ic_lora_video_only.py       # 영상 전용 IC-LoRA 파이프라인 스크립트
│   ├── ic_lora_video_only.provenance.json
│   └── check_gizmo_structure.py    # 기즈모 Tcl 문법 및 노드 연결 헤드리스 검증기
│
├── tests/                          # 100% 헤드리스 유닛 테스트 스위트
│   ├── test_matte.py               # 코어/엣지 수학 및 형태학 알고리즘 검증
│   ├── test_vlm.py                 # VLM 좌표 변환 및 SAM2 추론 검증
│   └── test_worker.py              # 서브프로세스 IPC, 프로세스 트리 강제 종료 및 청크 생성 검증
│
├── experiments/                    # 벤치마크 및 실험 데이터
│   ├── 001/                        # 기본 알파 프롬프트 대조군 데이터
│   └── 002/                        # 공간 어텐션 마스크 조건화 검증 데이터
│
└── jobs/                           # 비동기 생성 잡 작업 디렉토리
    ├── demo_refine/                # 데모 실행용 캐시 데이터
    └── gpu.lock                    # 동시 GPU 접근 방지 락 파일
```

---

## 🎛️ Nuke 기즈모 단일 탭 구성 (Unified Linear Workflow)

기존의 번거로운 탭 이동과 접이식 토글을 전면 제거하고, 각 단계 사이에 **시각적 네이티브 구분선**을 배치하여 위에서부터 차례대로 조작할 수 있는 **직관적인 단일 탭**으로 통합했습니다:

1. **Step 1. TARGET SELECTION (VLM & SAM2)**:
   - **Target Prompt**: 자연어 프롬프트(예: `"중앙의 갈색 셔츠를 입은 인물"`) 입력
   - **Ref Frame**: 대상이 가장 뚜렷한 기준 프레임 (기본값: `1`)
   - **Mask Expansion**: SAM 분할 마스크 외곽 자동 확장 비율 (기본값: `1.2배`, 피사체 움직임/헤어/모션 블러 여유 공간 확보)
   - **[🔍 Analyze Target (VLM)]**: 1~2초 내에 BBox 그라운딩 및 SAM2 마스크 추출 후 1.2배 자동 확장 (완료 시 뷰어 자동 전환)
2. **Step 2. LTX ALPHA VIDEO GENERATION**:
   - **First / Last**: 생성할 프레임 구간 설정 (기존 121프레임 한계를 해제하고 **49프레임 청크 분할 및 17프레임 코사인 오버랩 파이프라인**을 적용하여 수백~수천 프레임 롱테이크도 VRAM 초과 없이 끊김 없이 생성)
   - **[🚀 Generate Alpha Video]**: 백그라운드 LTX 비디오 디퓨전 생성 실행
   - **[⏹ Stop Queue (중단)]**: 백그라운드에서 실행 중인 큐 및 GPU/CPU 서브프로세스 트리(`taskkill /T /F`)를 즉시 강제 종료하고 VRAM 완전 회수
   - **실시간 프로그레스 바**: `[████████░░░░] 65% (LTX Denoising step 5/8)` 실시간 표시 및 작업 완료 시 결과 자동 연결
3. **Step 3. MATTE REFINEMENT (CORE & EDGE)**:
   - 노드 프로퍼티에 직접 노출된 슬라이더로 즉시 조절:
     - `Density & Alpha Curve`: Background Cutoff, White Level, Alpha Gamma
     - `Core Solidification`: Core Threshold, Inset (px), Strength, Fill Small Holes
     - `Edge & Detail Restoration`: Safe Radius (px), Edge Restoration, Edge Feather
   - **[⚡ Update Matte (CPU)]**: GPU 재실행 없이 CPU에서 밀리초 단위로 결과 즉시 갱신
4. **Step 4. OUTPUT & VIEW MODE**:
   - 7개 뷰 모드 실시간 전환 (`Final Alpha`, `Raw LTX Alpha`, `Target Selection (VLM)`, `Core`, `Edge`, `Source with Alpha`, `Premultiplied RGBA`)

---

## 👁️ 뷰 모드 (View Modes)

- `Final alpha`: 코어 채움, 엣지 복원, CoreAdd/Garbage가 모두 적용된 최종 완성 알파.
- `Raw LTX alpha`: LTX 신경망이 생성한 순수 원시 알파.
- `Target Selection (VLM)`: VLM + SAM2가 감지한 1프레임 타겟 마스크 (**생성 전 사전 검증용**).
- `Core`: 침식 및 구멍 메우기로 확보한 고체 코어 매트.
- `Edge`: 가우시안 완화 및 고주파 디테일이 보존된 부드러운 엣지 매트.
- `Source with alpha`: 원본 RGB에 알파 채널을 주입한 상태.
- `Premultiplied RGBA`: 합성용 프리멀티플라이 결과.

---

## 📥 설치 및 설정 가이드 (Installation Guide)

AutoRoto와 동일하게 Nuke의 표준 플러그인 디렉토리(`~/.nuke/`)에 배치하여 영구 설치하거나, 즉시 실행할 수 있습니다.

### 1. Nuke 영구 설치 (권장)

Nuke가 시작될 때마다 자동으로 AlphaGen 메뉴와 노드가 로드되도록 설정합니다.

1. 사용자의 `.nuke` 디렉토리에 있는 **`init.py`** 파일(`C:\Users\<사용자명>\.nuke\init.py`)을 열고 아래 코드를 추가합니다:

   ```python
   # ~/.nuke/init.py
   import nuke
   nuke.pluginAddPath('./AlphaGen')
   ```

   *(또는 절대 경로: `nuke.pluginAddPath(r"C:\Users\<사용자명>\.nuke\AlphaGen")`)*

2. 이제 Nuke를 실행하면:
   - [init.py](init.py)가 자동으로 기즈모(`gizmos/`) 및 파이썬 모듈 경로를 등록합니다.
   - [menu.py](menu.py)가 Nuke 좌측 노드 툴바(Nodes Toolbar)에 **✦ AlphaGen** 메뉴 아이콘을 자동 생성합니다.
   - 툴바 클릭 한 번으로 `AlphaGen` 기즈모를 바로 생성할 수 있습니다.

> [!TIP]
> **단축키 지정 (선택 사항)**:
> `~/.nuke/menu.py`에 단축키(예: `Ctrl+Alt+A`)를 등록하여 더욱 빠르게 생성할 수 있습니다:
>
> ```python
> # ~/.nuke/menu.py
> toolbar = nuke.menu("Nodes")
> m = toolbar.findItem("AlphaGen") or toolbar.addMenu("AlphaGen")
> m.addCommand("AlphaGen", "import alphagen_nuke; alphagen_nuke.create()", "ctrl+alt+a")
> ```

---

### 2. 의존성 패키지 설치 (Requirements)

기존 파이썬 환경(시스템 Python 또는 기존 AI 런타임)에서 AlphaGen 폴더로 이동 후 패키지를 설치합니다:

```powershell
cd C:\Users\<사용자명>\.nuke\AlphaGen
pip install -r requirements.txt
```

> [!TIP]
> **PyTorch 환경 재사용**:
> 이미 PC에 PyTorch와 CUDA가 설치되어 있거나 ComfyUI 등의 AI 환경(`D:/AI/ComfyUI_DATA/.venv` 등)이 있다면 별도로 가상환경을 만들 필요 없이 위 명령어만 실행하면 즉시 준비가 완료됩니다.

주요 패키지 요약:
- **Deep Learning**: `torch>=2.1.0`, `torchvision>=0.16.0` (CUDA 지원 필수)
- **Vision & SAM2 Models**: `transformers>=4.45.0` (Sam2Model 지원 버전), `accelerate>=0.26.0`, `safetensors>=0.4.0`
- **Video & Image I/O**: `opencv-python>=4.8.0` (무손실 FFV1 코덱 지원), `Pillow>=10.0.0`, `numpy>=1.24.0`
- **API & Networking**: `requests>=2.28.0`

---

### 3. 환경 변수 설정 (Environment Variables)

AlphaGen이 외부 API 및 GPU 런타임을 올바르게 찾을 수 있도록 환경 변수를 설정합니다.

| 환경 변수명 | 필수 여부 | 설명 | 설정 예시 |
| :--- | :---: | :--- | :--- |
| **`GEMINI_API_KEY`** | **필수** | VLM 공간 객체 그라운딩(BBox 추출)을 위한 Google AI Studio API 키 | `AIzaSyD...` |
| **`ALPHAGEN_PYTHON`** | 선택 | AI 백엔드 Python 실행기 경로 (미설정 시 `ComfyUI` venv 또는 시스템 python 자동 탐색) | `D:\AI\ComfyUI_DATA\.venv\Scripts\python.exe` |
| **`ALPHAGEN_MODELS_DIR`** | 선택 | LTX 모델 가중치가 위치한 기본 디렉토리 경로 (미설정 시 드라이브 자동 탐색) | `D:\AI\Models` |
| **`ALPHAGEN_CACHE_DIR`** | 선택 | 임시 연산 작업 캐시 경로 (미설정 시 Nuke `Preferences.localCachePath` / `NUKE_TEMP_DIR` / 시스템 임시 폴더 자동 연동) | `D:\NukeCache\AlphaGen` |

> [!NOTE]
> **API 키 설정 방법**:
>
> 1. [Google AI Studio](https://aistudio.google.com/)에서 무료 API 키를 발급받습니다.
> 2. Windows 시스템 환경 변수에 등록하거나, `~/.nuke/init.py` 상단에 직접 추가합니다:
>
>    ```python
>    import os
>    os.environ["GEMINI_API_KEY"] = "AIzaSy..."
>    ```

---

### 4. 설치 검증 (Verification & Self-Test)

터미널에서 유닛 테스트를 실행하여 환경 및 엔진이 정상 작동하는지 확인합니다:

```powershell
python -m unittest discover tests
# 결과: Ran 21 tests ... OK
```

---

## 📁 생성된 마스크 및 결과 파일 저장 위치 (File Storage)

생성된 모든 알파 마스크는 아티스트가 Nuke 및 타 합성 툴에서 바로 연결하여 사용할 수 있도록, **원본 영상 파일이 위치한 디렉토리 하위에 `alpha/` 폴더를 자동 생성**하여 저장됩니다. 디렉토리 오염을 방지하고 작업 편의성을 극대화하기 위해 꼭 필요한 **`final/`과 `raw/` 2개 폴더 체계**로 정돈되어 출력됩니다:

```text
[원본 영상 위치]/
├── source_plate.%04d.exr           # 원본 영상 시퀀스 (또는 clip.mp4)
│
└── alpha/                          # ★ 원본과 동일한 위치에 자동 생성되는 알파 폴더
    ├── final/
    │   └── alpha.%06d.png          # ★ [최종 완성 알파] 16-bit 무손실 완성 알파 시퀀스 (Nuke Read_final 자동 로드)
    ├── raw/
    │   └── alpha.%06d.png          # LTX 신경망이 생성한 원시 알파 시퀀스 (Nuke Read_raw 자동 로드)
    ├── vlm_initial_mask.png        # SAM2 1.2배 자동 확장 타겟 마스크 (사전 검증용)
    ├── vlm_selection.json          # VLM 바운딩 박스 및 신뢰도 메타데이터
    └── result.json                 # 최종 결과 명세서
```

> [!NOTE]
>
> - **깔끔한 산출물 관리 (raw & final 2폴더 체계)**: 사용자 프로젝트 디렉토리에는 불필요한 중간 디버그 파일 없이 최종 마스크(`final/`)와 원시 신경망 마스크(`raw/`)만 생성됩니다.
> - **Nuke 설정 캐시 경로 자동 연동**: 어텐션 비디오, 모델 텐서 캐시(`generic_alpha.decoded.pt`), 코어/엣지 분리 매트 등 수 기가바이트의 무거운 작업 임시 파일은 플러그인 폴더를 차지하지 않고 **Nuke의 공식 캐시 디렉토리(`Preferences.localCachePath` / `NUKE_TEMP_DIR` / `%TEMP%/nuke/AlphaGen/<job_id>/`)**로 자동 분리 격리됩니다.
> - **Nuke 뷰어 자동 연동**: 마스크 생성이 완료되면 Nuke 기즈모 내부의 `Read_final`, `Read_raw`, `Read_selection` 노드가 `<원본폴더>/alpha/` 경로를, `Read_core`와 `Read_edge`는 Nuke 캐시 경로를 **100% 자동으로 연결하여 실시간 표시**합니다.
> - **Windows 파일 잠금 방어**: Nuke 백그라운드 폴링과 작업 프로세스 간 파일 충돌(`WinError 5/32`)을 방지하기 위한 재시도 및 안전 쓰기 로직이 적용되어 있습니다.
> - 원본 소스가 파일이 아닌 가상 노드(Constant 등)인 경우 Nuke 캐시 디렉토리 하위에 안전하게 자동 폴백 저장됩니다.

---

## ⚙️ 하드웨어 및 모델 사양

- **권장 GPU**: NVIDIA RTX 4080 (16GB VRAM) 이상
- **런타임**: Python 3.10 ~ 3.12 (검증 완료: Python 3.12, PyTorch 2.10+cu130)
- **기반 모델 파일 (기본 탐색 경로: `<MODELS_DIR>/`)**:
  - `LTX-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors`
  - `LTX-2.5/vae/ltx-2.5-video-vae-bf16.safetensors`
  - `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`
  - `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`
  - `ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors`
- **VLM & 분할 가중치**:
  - Gemini Flash API (`GEMINI_API_KEY`)
  - HuggingFace 캐시 로컬 SAM2.1 (`facebook/sam2.1-hiera-large`)
