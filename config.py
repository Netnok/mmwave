# config.py

# 1. 모니터링할 '시행 폴더' 경로
#    (예: 'H1/1' 또는 'H2/6' 등)
TARGET_FOLDER = r'C:\Users\woori\Downloads\executive\dataset\H4\15'

# 2. 모델 및 통계 파일 경로
MODEL_PATH = r'C:\Users\woori\Downloads\executive\dataset\my_gru_window_model.pth'      # 👈 우리가 학습한 모델
STATS_PATH = r'C:\Users\woori\Downloads\executive\dataset\normalization_stats.json' # 👈 학습 시 생성된 통계

# --- (선택) GUI 및 모델 설정 ---

# 3. 모델 입력 파라미터 (학습 때와 동일해야 함)
WINDOW_SIZE = 10
INPUT_FEATURES = 15

# 4. GUI 설정
APP_TITLE = "실시간 HR 예측 (GRU)"
POLL_INTERVAL_MS = 1000  # 1초마다 폴더를 다시 스캔
GRAPH_HISTORY_SIZE = 100 # 그래프에 표시할 최근 데이터 개수