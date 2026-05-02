# GRS-ANFIS

`notebooks/maincode_with_ph.ipynb`를 중심으로 다시 묶은 독립 실행용 프로젝트입니다.

## 포함 범위
- 메인 노트북: `notebooks/maincode_with_ph.ipynb`
- boundary ablation 노트북: `notebooks/ablation2_complementary_boundary.ipynb`
- boundary ablation 모듈: `ablation2_complementary_boundary.py`
- 핵심 코드: `model.py`, `learning.py`, `data.py`, `utils.py`, `gh_eval_utils.py`, `interpretability.py`, `feature_schema.py`, `gh_config.py`
- 하이퍼파라미터: `hyper_parameter/best_*.json`, `hyper_parameter/ph_experiment_summary.csv`

## 별도 관리 범위
- 원자료 스냅샷: `data/`
- fold별 학습 가중치와 SVM estimator: `hyper_parameter/cv_weights/`, `*.pt`, `*.joblib`
- 실행 산출물: `output/`, `outputs/`

위 항목은 `.gitignore`로 제외되어 GitHub 저장소에 포함되지 않습니다. 필요한 파일 목록은 `data/README.md`를 참고해 로컬에서 별도로 배치합니다.

## 정리 원칙
- 원본 `GH-ANFIS_exp`는 수정하지 않고 유지합니다.
- 새 폴더는 노트북 실행에 필요한 최소 핵심 파일만 복사했습니다.
- `data.py`는 로컬 스냅샷이 있으면 우선 사용하고, 일부 UCI/OpenML 데이터는 없을 때 원격 fetch를 시도합니다.
- `hyper_parameter/cv_weights/`, `outputs/`, `output/`는 로컬 실행 결과 저장용 디렉터리입니다.

## 전처리 정책
- categorical column만 one-hot encoding 합니다.
- integer-valued ordinal / discrete numerical column은 기본적으로 numeric 그대로 유지합니다.
- 따라서 BCWD의 1~10 점수형 변수는 더 이상 one-hot으로 확장되지 않습니다.
- Vowel처럼 실제 categorical column(`Speaker_Number`, `Sex`)이 포함된 데이터셋은 one-hot encoding이 적용됩니다.

현재 loader 기준 예상 입력 차원은 다음과 같습니다.

| Dataset | Raw inputs | Training inputs |
|---|---:|---:|
| `Breast_Cancer_Wisconsin_(Original)` | 9 | 9 |
| `Vowel` | 12 | 26 |
| `Spambase` | 57 | 57 |
| `Gisette` | 5000 | 5000 |

## 실행 방법
1. 이 폴더를 프로젝트 루트로 사용합니다.
2. 필요한 패키지를 설치합니다.
3. 별도 보관한 데이터 스냅샷을 `data/` 아래에 배치합니다.
4. 기존 fold 가중치가 필요한 분석은 `hyper_parameter/cv_weights/` 아래에 배치합니다.
5. `notebooks/maincode_with_ph.ipynb`를 열어 순서대로 실행합니다.
6. boundary Q1 분석은 `notebooks/ablation2_complementary_boundary.ipynb`에서 실행합니다.

## 생성되는 결과
- fold별 저장 가중치: `hyper_parameter/cv_weights/`
- 요약 CSV: `outputs/maincode_with_ph_no_mi_summary.csv`
- boundary ablation 결과: `output/ablation2_complementary_boundary/`

## 참고
- 데이터셋 스냅샷과 학습 가중치는 저장소에서 제외되어 있으므로, 재현 실행 전에 로컬에 별도로 복원해야 합니다.
- 노트북 첫 셀은 `GH-ANFIS_E316`와 기존 `GH-ANFIS_exp` 둘 다 자동 인식하도록 정리했습니다.
- boundary ablation은 노트북에서 `run_ablation(...)`으로 실행되며, 기본 설정은 `no_mi` 가중치와 `boundary_bin=1 (Q1)`입니다.
- 노트북 기본 설정에서는 `summary_check`를 비활성화해 바로 실행되도록 했고, 필요하면 `summary_check_path`에 CSV 경로를 넣어 기존 결과와 비교할 수 있습니다.
- 과거 실행 결과가 저장된 노트북 output cell에는 이전 입력 차원(예: BCWD 80)이 남아 있을 수 있습니다. 변경된 전처리 결과는 노트북을 다시 실행해 반영해야 합니다.
