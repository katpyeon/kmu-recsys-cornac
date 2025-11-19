# Cornac 추천 시스템 프로젝트

추천 시스템 대회를 위한 Cornac 기반 모델 구현 프로젝트입니다.

---

## 1. 아나콘다 가상환경 생성

### 환경 생성 (처음 한 번만 실행)

```bash
# Python 3.12 환경 생성
conda create -n cornac python=3.12 -y

# 환경 활성화
conda activate cornac

# 패키지 설치
pip install -r requirements.txt
```

**이미 환경이 있다면** 환경 생성 단계는 건너뛰고 활성화부터 진행하세요.

---

## 2. 가상환경 활성화/비활성화

### 활성화
```bash
conda activate cornac
```

### 비활성화
```bash
conda deactivate
```

---

## 3. 가상환경 제거

```bash
# 환경 제거 (주의: 되돌릴 수 없습니다)
conda env remove -n cornac
```

---

## 4. 포함된 모델

| 노트북 | 모델명 | 설명 | 예상 Public LB | 학습 시간 |
|--------|--------|------|----------------|-----------|
| **01-ensemble** | EASE + BPR + ItemKNN | RRF 앙상블 (최고 성능) | **0.10-0.13** | ~30분 |
| **02-EASE** | EASE | 선형 오토인코더, 빠른 학습 | 0.08-0.10 | ~5분 |
| **03-BPR** | BPR | Pairwise ranking, 최고 단일 모델 | **0.09-0.12** | ~5분 |
| **04-ItemKNN** | ItemKNN | Item-item 유사도 기반 (앙상블용) | 0.05-0.06 | ~7분 |
| **05-BiVAECF** | BiVAECF | Variational autoencoder (실험용) | 0.02-0.03 | ~15분 |

### 모델 특징

#### **01-ensemble (추천)** ⭐
- **3개 모델 앙상블**: EASE (global pattern) + BPR (ranking) + ItemKNN (local similarity)
- **RRF 앙상블**: Reciprocal Rank Fusion으로 순위 통합
- **최적화**: Optuna TPE 하이퍼파라미터 튜닝
- **성능**: 가장 높은 Public LB 예상

#### **02-EASE**
- **알고리즘**: Closed-form 선형 오토인코더
- **장점**: 매우 빠른 학습 (~5분), 해석 가능
- **최적 λ**: 100 (Validation Recall@5: 0.4058)

#### **03-BPR** ⭐
- **알고리즘**: Bayesian Personalized Ranking (Matrix Factorization)
- **장점**: 암묵적 피드백에 최적화, 단일 모델 중 최고 성능
- **최적 파라미터**: k=130, lr=0.048, λ=0.0021

#### **04-ItemKNN**
- **알고리즘**: Cosine 유사도 기반 Item-item CF
- **목적**: 앙상블 다양성 제공 (단독 사용 비추천)
- **고정 파라미터**: k=50

#### **05-BiVAECF** ⚠️
- **알고리즘**: Bilateral Variational Autoencoder
- **주의**: 99.9% 희소 데이터에서 성능 매우 낮음 (Recall@5: 0.03)
- **용도**: 실험 및 학습 목적으로만 추천

---

## 5. 데이터 경로

### 입력 데이터
```
datasets/apply_train.csv
```

**데이터 형식:**
- `resume_seq`: 사용자 ID
- `recruitment_seq`: 채용공고 ID
- (노트북에서 자동으로 `rating=1` 추가)

### 출력 데이터 (제출 파일)
```
outputs/{YYYY-MM-DD}/submit_{모델명}_{YYYYMMDDhhmmss}.csv
```

**출력 예시:**
```
outputs/2025-11-19/submit_Ensemble_EASE_BPR_ItemKNN_20251119143025.csv
outputs/2025-11-19/submit_BPR_k130_20251119143530.csv
outputs/2025-11-19/submit_EASE_lambda100_20251119144015.csv
```

**참고:**
- `datasets/` 디렉토리는 수동으로 생성 필요
- `outputs/` 디렉토리는 노트북 실행 시 자동 생성
- 제출 파일은 날짜별로 자동 분류됨

---
