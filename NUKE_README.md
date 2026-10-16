# AlphaGen Nuke 기즈모 0.3 — 단일 탭 직관형 워크플로우 & 실시간 프로그레스 바

자연어 프롬프트(VLM) 기반의 객체 자동 타겟팅, SAM2 분할 마스크 생성, LTX Video Diffusion 어텐션 컨디셔닝, CompMatte 코어/에지 정밀 보정을 결합한 완성형 Nuke 알파 생성 기즈모입니다.

---

## 🚀 주요 개선 사항 (v0.3)

1. **단일 탭(Single-Tab) 통합 & 선형 직관형 배치**:
   - 탭 이동과 접이식 토글(Toggle Groups)을 전면 제거하고, 아티스트의 작업 순서에 맞춰 **위에서부터 차례대로(Step 1 $\to$ Step 4) 조작할 수 있는 직관적인 1개 탭**으로 통합했습니다.
2. **실시간 백그라운드 프로그레스 바 (`ag_progress_bar`)**:
   - VLM 분석, LTX 비디오 디퓨전 디노이징(Denoising Steps), CPU 매트 정제 과정이 **노드 프로퍼티 상에서 컬러 그래디언트 프로그레스 바(%)와 상태 메시지로 실시간 표시**됩니다.
3. **수동 새로고침 버튼 제거 및 100% 자동 로딩**:
   - 작업 완료 시 백그라운드 타이머가 감지하여 **최종 결과 및 VLM 프리뷰 마스크를 뷰어에 자동으로 즉시 로드**합니다. (수동 Refresh/Load 불필요)
4. **개인 절대 경로 완전 은닉 & 동적 자동 감지**:
   - 하드코딩된 드라이브/폴더 절대 경로를 완전히 제거하고, 환경 변수(`ALPHAGEN_MODELS_DIR`, `ALPHAGEN_PYTHON`) 또는 상대 경로 자동 탐색을 통해 이식성과 보안성을 확보했습니다.

---

## 🛠️ Nuke 설치 및 등록 (Installation)

AutoRoto와 동일하게 `~/.nuke/init.py`에 경로를 한 줄 추가하면 Nuke 실행 시 좌측 노드 툴바에 메뉴가 자동 등록됩니다:
```python
# C:\Users\<사용자명>\.nuke\init.py 에 추가
import nuke
nuke.pluginAddPath('./AlphaGen')
```
* **단축키 등록 (선택)**: `~/.nuke/menu.py`에 추가:
  ```python
  toolbar = nuke.menu("Nodes")
  m = toolbar.findItem("AlphaGen") or toolbar.addMenu("AlphaGen")
  m.addCommand("AlphaGen", "import alphagen_nuke; alphagen_nuke.create()", "ctrl+alt+a")
  ```

---

## 📋 단일 탭 순서별 워크플로우 (Step-by-Step)

### Step 1. TARGET SELECTION (VLM & SAM2)
* **Target Prompt**: 분리하고자 하는 대상을 자연어로 입력합니다. (예: `중앙의 갈색 셔츠를 입은 인물`, `red sports car`)
* **Ref Frame**: 대상이 가장 잘 보이는 기준 프레임 (기본값: `1`)
* **Mask Expansion**: SAM2 마스크 외곽 자동 확장 비율 (기본값: `1.2배`, 피사체 움직임/헤어/모션 블러 여유 공간 확보)
* **[🔍 Analyze Target (VLM)]**: 1~2초 내에 Gemini Flash가 BBox를 그라운딩하고, SAM2가 픽셀 단위 마스크를 추출한 뒤 1.2배 자동 확장합니다.
* 완료 시 뷰어가 자동으로 `Target Selection (VLM)`으로 전환되어 생성 전 마스크를 즉시 검증할 수 있습니다.

### Step 2. LTX ALPHA VIDEO GENERATION
* **First / Last**: 생성할 프레임 구간 설정 (8n+1 배수 규칙 자동 준수)
* **Working Max Side**: 작업 해상도 (기본값: `512`)
* **[🚀 Generate Alpha Video]**: LTX 비디오 디퓨전 생성을 백그라운드에서 시작합니다.
* **실시간 프로그레스 바**: `[████████░░░░] 65% (LTX Denoising step 5/8)` 형태로 실시간 진행률이 표시되며, 완료 시 결과가 자동으로 연결됩니다.

### Step 3. MATTE REFINEMENT (CORE & EDGE)
* 토글 없이 노드 상에 직접 슬라이더가 노출되어 있습니다:
  - **Density & Alpha Curve**: Background Cutoff, White Level, Alpha Gamma
  - **Core Solidification**: Core Threshold, Inset (px), Strength, Fill Small Holes
  - **Edge & Detail Restoration**: Safe Radius (px), Edge Restoration, Edge Feather
* **[⚡ Update Matte (CPU)]**: GPU 재실행 없이 CPU에서 밀리초 단위로 결과를 즉시 갱신합니다.

### Step 4. OUTPUT & VIEW MODE
* `view_mode` 드롭다운에서 원하는 뷰를 선택합니다:
  - `Final alpha`: 코어 채움, 엣지 복원, CoreAdd/Garbage가 모두 적용된 최종 완성 알파.
  - `Raw LTX alpha`: LTX 신경망이 순수하게 예측한 원시 알파.
  - `Target Selection (VLM)`: VLM + SAM2가 감지한 1프레임 타겟 마스크 (**사전 검증용**).
  - `Core`: 침식 및 구멍 메우기로 확보한 고체 코어 매트.
  - `Edge`: 가우시안 완화 및 고주파 디테일이 보존된 부드러운 엣지 매트.
  - `Source with alpha`: 원본 RGB에 알파 채널을 결합한 상태.
  - `Premultiplied RGBA`: 합성용 프리멀티플라이 결과.

---

## 📁 생성 파일 저장 경로 (Output Storage)

생성된 알파 마스크는 원본 영상이 위치한 폴더 하위에 **`alpha/`** 폴더가 자동 생성되어 저장됩니다:
* **최종 완성 알파 시퀀스**: `[원본폴더]/alpha/alpha.%06d.png` (Nuke Read_final에 자동 로드)
* **VLM & SAM2 타겟 마스크**: `[원본폴더]/alpha/vlm_initial_mask.png`
* **LTX 원시 알파 시퀀스**: `[원본폴더]/alpha/raw/alpha.%06d.png`
* **코어 / 엣지 분리 매트**: `[원본폴더]/alpha/core/`, `[원본폴더]/alpha/edge/`

*(작업 완료 시 Nuke 기즈모 내부 Read 노드들이 위 파일들을 100% 자동으로 읽어 들입니다. 원본이 파일이 아닌 가상 노드일 경우 `jobs/` 폴더에 안전하게 자동 저장됩니다.)*
