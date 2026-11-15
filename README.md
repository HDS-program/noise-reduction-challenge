# 내 목소리를 구해줘! - 음성 잡음 제거

- 팀명: 신호등 (10조)
- 제출 회차: 2회차
- 제출 버전: 태그 `submission-2b` (커밋 `39bc8a9`, 브랜치 `dongsik`)

## 방법
딥러닝·머신러닝을 쓰지 않는 STFT 기반 HRNR(Harmonic Regeneration Noise Reduction) 필터입니다. 처리 과정에서 Clean 음성은 사용하지 않고, 입력 파일 하나만 보고 동작합니다.

1. 150 Hz 고역통과 필터로 저역 잡음을 제거합니다.
2. 창 길이 1536(96 ms), hop 384로 STFT를 계산합니다.
3. 에너지가 낮은 프레임 10%의 평균으로 잡음 파워를 처음 추정합니다.
4. **주파수 칸마다 시간에 따른 파워 변동(변동계수 cv = 표준편차/평균)을 계산해 목소리 칸과 잡음 칸을 나눕니다.** 목소리가 있는 칸은 말할 때와 쉴 때의 차이가 커서 cv가 크고, 꾸준한 잡음이나 선 모양 잡음(사이렌 등)은 cv가 작습니다. cv < 1.3이고 처음 추정의 -3 dB 이상인 칸은 잡음 칸으로 보고, 잡음 파워를 파일 전체 평균으로 다시 잡은 뒤 이득 하한을 0.01로 낮춥니다. 나머지(목소리 칸)는 이득 하한 0.15를 씁니다.
5. decision-directed 방식(평활 0.95)으로 사전 SNR을 추정하고, 두 단계 SNR 추정(TSNR)과 배음 재생(HRNR, rho 0.5)으로 이득을 계산합니다.

파라미터는 Train 100개 파일의 PESQ로 고르고, 후보가 여러 개일 때는 Validation 164개로 최종 선택했습니다.

## 설치와 실행
```bash
pip install -r requirements.txt
python denoise.py --input input --output output
```

input 폴더의 모든 WAV 파일을 처리해 같은 파일명으로 output 폴더에 저장합니다. 샘플링 주파수, 채널 수, 샘플 수는 입력과 같습니다.

## Validation 결과 (164개 파일 평균)
| 방법 | SNR (dB) | PESQ-WB |
|---|---|---|
| Noisy 입력 | 9.6341 | 1.3947 |
| 조교 베이스라인 Wiener (기본 파라미터) | 13.1196 | 1.5984 |
| 1회차 제출 (태그 submission-1, Wiener 튜닝) | 13.4683 | 1.7121 |
| 2회차 후보 A (태그 submission-2, 신효찬 v2) | 8.0394 | 1.8881 |
| **2회차 제출 (태그 submission-2b, 이 버전)** | **8.3560** | **1.9656** |

SNR이 1회차보다 낮은 것은 150 Hz 고역통과가 음성의 가장 낮은 성분도 함께 줄이기 때문입니다. 순위 기준인 PESQ를 우선했습니다.

## 코드 출처
STFT 처리, 조용한 프레임 기반 잡음 추정, decision-directed 사전 SNR 추정의 기본 구조는 조교가 제공한 베이스라인 노트북의 구현을 기반으로 했습니다. TSNR·HRNR 이득은 Plapous, Marro, Scalart (2006)를 참고해 구현했고, 변동계수(cv) 기반 목소리/잡음 칸 분류와 파라미터는 Train·Validation 실험으로 직접 정했습니다. 150 Hz 고역통과와 창 길이 실험은 신효찬 v2의 결과를 참고했습니다.

## 테스트 환경
Google Colab (numpy, scipy, soundfile — requirements.txt 범위)
