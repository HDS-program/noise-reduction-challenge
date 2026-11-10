# 내 목소리를 구해줘! - 음성 잡음 제거 (Wiener Filter)

- 팀명: 신호등
- 제출 회차: 1
- 제출 버전: 브랜치 hyo-1213257, 태그 submission-1

## 방법
STFT 기반 Wiener Filter입니다. 딥러닝·머신러닝은 사용하지 않았고, 처리 과정에서 Clean 음성은 사용하지 않습니다.
잡음 파워는 파일 안에서 에너지가 낮은 프레임 하위 10%의 평균으로 추정하고, 이전 프레임 결과로 사전 SNR을 추정해 이득을 계산합니다.
파라미터(noise_fraction=0.1, wiener_smoothing=0.95, wiener_gain_floor=0.2)는 Train 100개 파일의 PESQ로 선택했습니다.

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
| 이 제출 (튜닝한 Wiener) | 13.4683 | 1.7121 |

## 코드 출처
STFT 처리, 잡음 파워 추정, Wiener 이득 계산은 조교가 제공한 베이스라인 노트북의 구현을 기반으로 했고, 파라미터만 변경했습니다.

## 테스트 환경
Google Colab, numpy 2.4.6 / scipy 1.16.3 / soundfile 0.14.0
