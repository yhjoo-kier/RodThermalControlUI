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

```bash
docker-compose up
```

- 백엔드: http://localhost:8000
- 프론트엔드: http://localhost:5173

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
