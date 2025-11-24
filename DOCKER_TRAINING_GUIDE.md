# Docker 환경에서 RL 학습 실행 가이드

Docker Compose로 실행 중인 환경에서 RL 학습을 수행하는 방법을 설명합니다.

## 방법 1: 웹 UI 사용 (가장 간단) ⭐

웹 UI에서 직접 학습을 시작할 수 있습니다.

1. 브라우저에서 `http://localhost:5173` 접속
2. 상단 네비게이션 바에서 **"🎓 Train RL Model"** 버튼 클릭
3. 학습 모달에서 설정 선택:
   - **Quick Presets** 버튼으로 빠른 설정
   - 또는 개별 파라미터 조정
4. 모달 하단의 버튼 클릭하여 학습 시작
5. 실시간으로 진행상황 모니터링

**장점:**
- GUI로 편리하게 설정 가능
- 실시간 진행상황 확인
- 백그라운드에서 실행되어 다른 작업 가능

## 방법 2: Docker Exec로 CLI 명령 실행

실행 중인 backend 컨테이너에서 직접 Python 명령 실행:

### 2-1. 기본 사용법

```bash
# 1. Docker 컨테이너가 실행 중인지 확인
docker-compose ps

# 2. backend 컨테이너에서 학습 실행
docker-compose exec backend python backend/train_rl.py --quick
```

### 2-2. 다양한 학습 옵션

```bash
# Quick Test (20k timesteps, ~2-5분)
docker-compose exec backend python backend/train_rl.py --quick --reward shaped

# Standard Training (500k timesteps, ~30-60분)
docker-compose exec backend python backend/train_rl.py --standard --reward shaped

# GPU Training (2M timesteps) - GPU 설정 필요
docker-compose exec backend python backend/train_rl.py --gpu --reward shaped --device cuda

# Custom Configuration
docker-compose exec backend python backend/train_rl.py \
  --timesteps 1000000 \
  --envs 12 \
  --reward shaped \
  --lr 0.0003 \
  --device auto
```

### 2-3. 실시간 로그 확인

별도 터미널에서 로그 확인:

```bash
# 터미널 1: 학습 실행
docker-compose exec backend python backend/train_rl.py --standard

# 터미널 2: 로그 스트리밍
docker-compose logs -f backend
```

## 방법 3: 일회성 컨테이너로 실행

학습만을 위한 별도 컨테이너를 실행:

```bash
# 일회성 컨테이너로 학습 실행
docker-compose run --rm backend python backend/train_rl.py --standard --reward shaped

# 백그라운드로 실행하고 로그 파일 저장
docker-compose run -d --rm backend python backend/train_rl.py --gpu > training.log 2>&1
```

**장점:**
- 메인 서비스와 독립적으로 실행
- 학습 완료 후 자동으로 컨테이너 제거 (--rm)

## 방법 4: GPU 지원 설정 (GPU가 있는 경우)

GPU를 사용하려면 docker-compose.yml을 수정해야 합니다.

### 4-1. docker-compose.yml 수정

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app/backend
      - ./ppo_thermal_rod.zip:/app/ppo_thermal_rod.zip
    environment:
      - PYTHONUNBUFFERED=1
    # GPU 설정 추가
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  frontend:
    image: node:18-alpine
    working_dir: /app
    volumes:
      - ./frontend:/app
    ports:
      - "5173:5173"
    command: sh -c "npm install && npm run dev -- --host"
```

### 4-2. nvidia-docker 확인

```bash
# NVIDIA Docker Runtime 확인
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 설치되지 않았다면
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### 4-3. GPU로 학습 실행

```bash
# docker-compose.yml 수정 후
docker-compose down
docker-compose up -d --build

# GPU 학습 실행
docker-compose exec backend python backend/train_rl.py --gpu --device cuda
```

## 학습 모델 관리

### 기존 모델 삭제 (새로 학습할 때)

```bash
# 호스트에서 삭제
rm ppo_thermal_rod.zip

# 또는 컨테이너 내부에서 삭제
docker-compose exec backend rm /app/ppo_thermal_rod.zip
```

### 학습된 모델 확인

```bash
# 모델 파일 확인
ls -lh ppo_thermal_rod.zip

# 컨테이너 내부에서 확인
docker-compose exec backend ls -lh /app/ppo_thermal_rod.zip
```

### 체크포인트 확인

```bash
# 학습 중 저장되는 체크포인트
docker-compose exec backend ls -lh /app/checkpoints/

# 호스트에서도 확인 가능 (볼륨 마운트 시)
ls -lh ./checkpoints/
```

## 학습 모니터링

### 방법 1: 웹 UI
- `http://localhost:5173`에서 실시간 진행상황 확인

### 방법 2: Docker 로그
```bash
# 실시간 로그 스트리밍
docker-compose logs -f backend

# 최근 100줄
docker-compose logs --tail=100 backend
```

### 방법 3: TensorBoard (선택사항)
```bash
# TensorBoard 실행 (포트 6006)
docker-compose exec backend tensorboard --logdir=/app/logs --host=0.0.0.0 --port=6006

# 브라우저에서 http://localhost:6006 접속
```

docker-compose.yml에 TensorBoard 서비스 추가:
```yaml
  tensorboard:
    image: tensorflow/tensorflow:latest
    ports:
      - "6006:6006"
    volumes:
      - ./logs:/logs
    command: tensorboard --logdir=/logs --host=0.0.0.0
```

## 학습 중단 및 재개

### 학습 중단

**웹 UI에서:**
- 학습 모달의 "Stop Training" 버튼 클릭

**CLI에서:**
- `Ctrl+C`로 프로세스 종료
- 자동으로 `ppo_thermal_rod_interrupted.zip` 저장됨

### 학습 재개

이전에 저장된 모델이 있으면 자동으로 로드하여 계속 학습:

```bash
# 모델이 있으면 자동으로 계속 학습
docker-compose exec backend python backend/train_rl.py --standard
```

## 트러블슈팅

### GPU 관련 오류
```bash
# CUDA를 찾을 수 없다는 오류 시
docker-compose exec backend python -c "import torch; print(torch.cuda.is_available())"

# False가 나오면 CPU로 학습
docker-compose exec backend python backend/train_rl.py --standard --device cpu
```

### 메모리 부족
```bash
# 병렬 환경 수 줄이기
docker-compose exec backend python backend/train_rl.py --timesteps 500000 --envs 2
```

### 컨테이너 재시작
```bash
# 변경사항 적용 후 재시작
docker-compose restart backend

# 완전히 재빌드
docker-compose down
docker-compose up -d --build
```

## 권장 워크플로우

### CPU 환경 (일반 개발)
```bash
# 1. 기존 모델 삭제
rm ppo_thermal_rod.zip

# 2. 컨테이너 실행
docker-compose up -d

# 3-1. 웹 UI에서 학습 (추천)
#      http://localhost:5173 접속 → Train RL Model 버튼

# 3-2. 또는 CLI로 학습
docker-compose exec backend python backend/train_rl.py --standard --reward shaped

# 4. 학습 완료 후 시뮬레이션에서 확인
#    RL Agent가 적극적으로 제어하는지 확인
```

### GPU 환경 (고성능 학습)
```bash
# 1. GPU 설정 확인
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 2. docker-compose.yml에 GPU 설정 추가 (위 참조)

# 3. 재빌드 및 실행
docker-compose down
docker-compose up -d --build

# 4. GPU 학습 실행
docker-compose exec backend python backend/train_rl.py --gpu --reward shaped --device cuda

# 5. 실시간 모니터링
docker-compose logs -f backend
```

## 학습 시간 참고

| 설정 | Timesteps | CPU 시간 | GPU 시간 |
|------|-----------|----------|----------|
| Quick | 20,000 | ~2-5분 | ~1-2분 |
| Standard | 500,000 | ~30-60분 | ~10-20분 |
| GPU | 2,000,000 | ~2-4시간 | ~30-60분 |
| Intensive | 10,000,000 | ~10-20시간 | ~2-4시간 |

*실제 시간은 하드웨어 사양에 따라 다를 수 있습니다*

## 추가 팁

1. **빠른 테스트**: 먼저 `--quick` 옵션으로 설정이 정상 작동하는지 확인
2. **점진적 학습**: 100k → 500k → 2M 순으로 점진적으로 학습량 증가
3. **로그 저장**: 긴 학습 시 로그 파일로 저장하여 나중에 분석
4. **모델 백업**: 좋은 성능의 모델은 별도 파일명으로 백업

```bash
# 모델 백업
cp ppo_thermal_rod.zip ppo_thermal_rod_backup_$(date +%Y%m%d_%H%M%S).zip
```
