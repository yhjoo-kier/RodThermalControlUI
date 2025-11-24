# Rod Thermal Control UI

열 로드 제어 시뮬레이션을 위한 풀스택 애플리케이션입니다. PID, MPC, RL 컨트롤러를 사용한 물리 기반 열 시뮬레이션과 3D 시각화 웹 UI를 제공합니다.

## 시스템 요구사항

- Python 3.11+
- Node.js 18+
- npm 또는 yarn

## 설치 방법

### 1. 백엔드 설정

```bash
cd backend
pip install -r requirements.txt
```

### 2. 프론트엔드 설정

```bash
cd frontend
npm install
```

## 실행 방법

### Option 1: Docker Compose 사용 (권장)

Docker와 Docker Compose가 설치되어 있다면:

#### 기본 명령어

```bash
# 첫 실행 또는 이미지 재빌드
docker-compose up --build

# 백그라운드에서 실행 (권장)
docker-compose up -d

# 로그 실시간 확인
docker-compose logs -f

# 특정 서비스 로그만 확인
docker-compose logs -f backend
docker-compose logs -f frontend

# 컨테이너 상태 확인
docker-compose ps

# 컨테이너 중지
docker-compose stop

# 컨테이너 중지 및 제거
docker-compose down

# 컨테이너 재시작
docker-compose restart
```

#### 접속 주소

- **프론트엔드**: http://localhost:5173
- **백엔드 API**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs

#### RL 모델 학습

Docker 환경에서 RL 학습을 실행하는 방법:

```bash
# 웹 UI에서 학습 (가장 간단)
# http://localhost:5173 접속 → "🎓 Train RL Model" 버튼 클릭

# 또는 CLI로 학습
docker-compose exec backend python backend/train_rl.py --quick      # 테스트 (2-5분)
docker-compose exec backend python backend/train_rl.py --standard   # 권장 (30-60분)
```

자세한 내용은 [DOCKER_TRAINING_GUIDE.md](./DOCKER_TRAINING_GUIDE.md) 참조

### Option 2: 로컬에서 직접 실행

#### 백엔드 실행

터미널 1:
```bash
# 프로젝트 루트 디렉토리에서 실행
export PYTHONPATH=/home/user/RodThermalControlUI:$PYTHONPATH
cd /home/user/RodThermalControlUI
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 프론트엔드 실행

터미널 2:
```bash
cd frontend
npm run dev
```

애플리케이션 접속: http://localhost:5173

## 프로젝트 구조

```
RodThermalControlUI/
├── backend/
│   ├── app/
│   │   └── main.py          # FastAPI 서버 및 WebSocket
│   ├── control/
│   │   ├── pid_controller.py
│   │   ├── mpc_controller.py
│   │   └── rl_agent.py
│   └── physics/
│       └── heat_equation.py  # 열 전달 물리 시뮬레이션
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Dashboard.jsx  # 메인 대시보드
│       │   └── Rod3D.jsx      # 3D 시각화
│       └── App.jsx
└── docker-compose.yml
```

## 기능

- **실시간 열 시뮬레이션**: 1D 열 전달 방정식 기반
- **다중 제어 알고리즘**: PID, MPC(Model Predictive Control), RL(Reinforcement Learning)
- **3D 시각화**: Three.js를 사용한 실시간 온도 분포 표시
- **실시간 차트**: 온도 및 제어 입력 히스토리
- **WebSocket 통신**: 실시간 데이터 스트리밍

## API 엔드포인트

- `GET /`: API 상태 확인
- `WS /ws`: WebSocket 연결 (시뮬레이션 데이터 스트리밍)

## 검증 스크립트

프로젝트에는 시스템 검증을 위한 스크립트들이 포함되어 있습니다:

- `verify_physics.py`: 물리 시뮬레이션 검증
- `verify_control.py`: 제어 알고리즘 검증
- `verify_ws.py`: WebSocket 연결 검증
- `verify_rl.py`: 사전 학습된 RL 정책이 0이 아닌 제어 입력을 생성하는지 확인
- `reproduce_mpc.py`: MPC 성능 분석
