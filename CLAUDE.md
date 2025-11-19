# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

Cornac 기반 추천 시스템 대회 프로젝트입니다. 암묵적 피드백(implicit feedback)을 사용하여 사용자-채용공고 추천을 수행합니다.

**데이터 특성:**
- 사용자 수: ~8,500명
- 아이템 수: ~6,700개
- 희소성: 99.9%
- 데이터 형식: `resume_seq` (사용자 ID), `recruitment_seq` (아이템 ID)

## 환경 설정

### 필수 패키지 설치
```bash
# Conda 환경 생성 및 활성화
conda create -n cornac python=3.12 -y
conda activate cornac

# 패키지 설치
pip install -r requirements.txt
```

### 주요 의존성
- `cornac>=2.3.0`: 추천 시스템 프레임워크
- `optuna>=3.6.0`: 하이퍼파라미터 최적화
- `torch>=2.1.0`: PyTorch (CUDA/MPS 가속 지원)
- `pandas`, `numpy`, `scikit-learn`: 데이터 처리

## 데이터 구조

### 입력 데이터
- 경로: `datasets/apply_train.csv`
- 형식: CSV (열: `resume_seq`, `recruitment_seq`)
- **중요**: `datasets/` 디렉토리는 수동으로 생성 필요

### 출력 데이터
- 경로: `outputs/{YYYY-MM-DD}/submit_{모델명}_{타임스탬프}.csv`
- 형식: CSV (열: `resume_seq`, `recruitment_seq`)
- 각 사용자당 정확히 5개의 추천 아이템
- **자동 생성**: `outputs/` 디렉토리와 날짜별 하위 디렉토리는 노트북 실행 시 자동 생성됨

## 구현된 모델

### 1. 앙상블 모델 (01-ensemble) - **최고 성능** ⭐
**파일**: `01-cornac-ensemble_EASE_BPR_ItemKNN.ipynb`

**구성**:
- EASE (lambda=100) + BPR (k=130) + ItemKNN (k=50)
- RRF (Reciprocal Rank Fusion) 앙상블
- Optuna TPE 하이퍼파라미터 최적화 (30 trials)

**예상 성능**:
- Validation Recall@5: 0.40-0.50
- Public LB: 0.10-0.13

**실행 시간**: ~30분

### 2. EASE (02-EASE)
**파일**: `02-cornac-EASE.ipynb`

**알고리즘**: Embarrassingly Shallow Autoencoders (Steck, 2019)
- Closed-form 선형 오토인코더
- Lambda 그리드 서치 (논문 기반)

**하이퍼파라미터**:
- `lamb`: 100-2000 범위 탐색 (최적: 100)

**모드**:
- `TUNING_MODE = 'fast'`: Lambda=100 고정 (~5분)
- `TUNING_MODE = 'thorough'`: 그리드 서치 7개 값 (~7분)

**예상 성능**: Validation Recall@5 ~0.40

### 3. BPR (03-BPR) - **최고 단일 모델** ⭐
**파일**: `03-cornac-BPR.ipynb`

**알고리즘**: Bayesian Personalized Ranking (Rendle et al., 2012)
- Pairwise ranking 최적화
- 암묵적 피드백 전용

**하이퍼파라미터** (검증된 최적값):
- `k`: 130 (latent factors)
- `learning_rate`: 0.0485
- `lambda_reg`: 0.0021
- `max_iter`: 250

**모드**:
- `TUNING_MODE = 'fast'`: 검증된 파라미터 사용 (~5분)
- `TUNING_MODE = 'thorough'`: Optuna TPE 30 trials (~20분)

**예상 성능**: Validation Recall@5 ~0.50

### 4. ItemKNN (04-ItemKNN) - **앙상블용**
**파일**: `04-cornac-ItemKNN.ipynb`

**알고리즘**: Item-based K-Nearest Neighbors (Sarwar et al., 2001)
- Cosine 유사도 기반

**하이퍼파라미터**:
- `k`: 50 (고정)
- `similarity`: 'cosine'

**용도**: 앙상블 다양성 제공 (단독 사용 비추천)

### 5. BiVAECF (05-BiVAECF) - **실험용** ⚠️
**파일**: `05-cornac-BiVAECF.ipynb`

**알고리즘**: Bilateral Variational Autoencoder
- 딥러닝 기반 변분 오토인코더

**주의사항**: 99.9% 희소 데이터에서 성능 매우 낮음 (Recall@5: 0.03)
**용도**: 실험 및 학습 목적으로만 권장

## 코드 아키텍처

### 공통 구조 (모든 노트북)

1. **설정 (Configuration)**
   - Device 자동 선택: CUDA > MPS > CPU
   - 데이터 컬럼명: `DEFAULT_USER_COL`, `DEFAULT_ITEM_COL`, `DEFAULT_RATING_COL`
   - 추천 개수: `TOP_K = 5`
   - Random seed: `SEED = 202511`
   - 튜닝 모드: `TUNING_MODE = 'fast'` 또는 `'thorough'`

2. **데이터 로딩**
   ```python
   DATA_FILE = 'datasets/apply_train.csv'  # 대회 당일 수정 필요
   data = pd.read_csv(DATA_FILE)
   data[DEFAULT_RATING_COL] = 1  # Implicit feedback
   ```

3. **Validation Split (thorough 모드)**
   - Temporal split: 사용자별 20% holdout
   - **중요**: Random split 사용 금지 (cold-start 문제 발생)
   - Cross-validation 사용 안 함 (과적합 방지)

4. **모델 학습**
   - Validation set으로 하이퍼파라미터 최적화 (thorough 모드)
   - 전체 데이터로 최종 모델 재학습

5. **Cold-start 처리**
   - 아이템 인기도 기반 폴백 메커니즘
   - Top-5 인기 아이템 자동 계산

6. **예측 생성**
   - 모든 사용자에 대해 Top-5 추천 생성
   - Cold-start 사용자: 인기도 폴백 적용

7. **제출 파일 생성**
   - 자동 디렉토리 생성: `outputs/{날짜}/`
   - 파일명 형식: `submit_{모델명}_{타임스탬프}.csv`

### 앙상블 전용 구조 (01-ensemble)

1. **EASE 그리드 서치**
   - Lambda 값: [100, 250, 500, 750, 1000, 1500, 2000]
   - Validation Recall@5 기준 최적값 선택

2. **BPR Optuna TPE**
   - 30 trials 하이퍼파라미터 탐색
   - 탐색 범위:
     - `k`: 80-150 (step=10)
     - `max_iter`: 150-250 (step=50)
     - `learning_rate`: 0.01-0.05 (log scale)
     - `lambda_reg`: 0.001-0.01 (log scale)

3. **ItemKNN 고정 파라미터**
   - `k=50`, `similarity='cosine'` (논문 권장값)

4. **RRF 앙상블**
   ```python
   def reciprocal_rank_fusion(model_predictions, k_constant=60):
       # RRF Score = Σ (1 / (k + rank_i))
       # 동일 가중치 (1/3 each model)
   ```

## 모델 선택 가이드

### 추천 시나리오

1. **최고 성능이 필요한 경우**:
   - `01-cornac-ensemble_EASE_BPR_ItemKNN.ipynb` 실행
   - 소요 시간: ~30분
   - 예상 LB: 0.10-0.13

2. **빠른 베이스라인이 필요한 경우**:
   - `03-cornac-BPR.ipynb` (fast 모드) 실행
   - 소요 시간: ~5분
   - 예상 LB: 0.09-0.12

3. **해석 가능한 모델이 필요한 경우**:
   - `02-cornac-EASE.ipynb` 실행
   - 선형 모델로 해석 가능
   - 소요 시간: ~5분

4. **하이퍼파라미터 재탐색이 필요한 경우**:
   - 각 노트북에서 `TUNING_MODE = 'thorough'` 설정
   - EASE: ~7분, BPR: ~20분

## 주요 주의사항

### 데이터 경로 변경
대회 당일 데이터 파일명이 변경되면:
```python
# 각 노트북의 "데이터 로딩" 셀에서 수정
DATA_FILE = 'datasets/apply_train.csv'  # ← 이 부분만 변경
```

### Validation 전략
- **절대 사용 금지**: Random split, K-Fold CV
- **권장**: Temporal split (사용자별 20% holdout)
- **이유**: Random split은 cold-start 문제로 Public LB 매우 낮음 (0.0055)

### Device 설정
- 자동 선택: CUDA > MPS > CPU
- BPR과 BiVAECF는 GPU/MPS 가속 지원
- EASE, ItemKNN은 CPU에서도 충분히 빠름

### 제출 파일 형식
- 각 사용자당 **정확히 5개** 추천 필요
- 열 이름: `resume_seq`, `recruitment_seq`
- Cold-start 사용자도 반드시 포함 (인기도 폴백 사용)

## 논문 참조

- **EASE**: Steck, 2019 - "Embarrassingly Shallow Autoencoders for Sparse Data" (WWW)
- **BPR**: Rendle et al., 2012 - "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI)
- **ItemKNN**: Sarwar et al., 2001 - "Item-based Collaborative Filtering Recommendation Algorithms" (WWW)
- **RRF**: Cormack et al., 2009 - "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" (SIGIR)

## 성능 벤치마크

| 모델 | Validation Recall@5 | 예상 Public LB | 학습 시간 |
|------|---------------------|----------------|-----------|
| Ensemble | 0.40-0.50 | 0.10-0.13 | ~30분 |
| BPR | ~0.50 | 0.09-0.12 | ~5분 |
| EASE | ~0.40 | 0.08-0.10 | ~5분 |
| ItemKNN | ~0.30 | 0.05-0.06 | ~7분 |
| BiVAECF | ~0.03 | 0.02-0.03 | ~15분 |

**참고**: Validation 성능과 Public LB는 데이터 분포 차이로 직접 비교 불가
