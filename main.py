# main.py

import os
import glob
import json
import time
import threading
import queue
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- 로컬 파일 임포트 ---
import config

# --- 1. GRU 모델 정의 (학습 때 사용한 것과 동일한 구조) ---
# 이 파일에 모델 구조가 정의되어 있어야 .pth 파일을 불러올 수 있습니다.
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)

# --- 2. 실시간 GUI 애플리케이션 ---
class RealTimeHRApp:
    def __init__(self, root):
        self.root = root
        self.root.title(config.APP_TITLE)
        self.root.geometry("800x600")

        # 모델 예측값을 GUI 스레드로 전달하기 위한 큐
        self.data_queue = queue.Queue()
        
        # 중복 예측을 막기 위해 마지막으로 처리한 '파형 개수'를 저장
        self.last_waveform_count = 0
        
        # 그래프 데이터
        self.plot_data = [np.nan] * config.GRAPH_HISTORY_SIZE

        # --- GUI 레이아웃 설정 ---
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 상태 표시줄
        self.status_label = ttk.Label(self.main_frame, text="초기화 중...", font=("Helvetica", 14))
        self.status_label.pack(pady=10)

        # 그래프 영역
        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("real-time (HR) prediction")
        self.ax.set_ylabel("Heart Rate (BPM)")
        self.ax.set_ylim(40, 120) # Y축 범위 고정 (필요시 수정)
        
        self.line, = self.ax.plot(self.plot_data, animated=True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

        # --- 3. 모델 및 정규화 통계 로드 ---
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 1. 정규화 통계 로드
            with open(config.STATS_PATH, 'r') as f:
                stats = json.load(f)
                self.mean = np.array(stats['mean']).reshape(1, config.INPUT_FEATURES)
                self.std = np.array(stats['std']).reshape(1, config.INPUT_FEATURES)
                # 0으로 나누기 방지
                self.std[self.std == 0] = 1.0

            # 2. 모델 구조 정의 및 가중치 로드
            # (모델 파라미터는 config.py가 아닌 학습 코드와 동일하게 맞춰야 함)
            self.model = GRUModel(
                input_size=config.INPUT_FEATURES, 
                hidden_size=64,  # 학습 때 사용한 hidden_size
                num_layers=2,    # 학습 때 사용한 num_layers
                output_size=1
            ).to(self.device)
            
            self.model.load_state_dict(torch.load(config.MODEL_PATH, map_location=self.device))
            self.model.eval() # 예측 모드로 설정

            self.status_label.config(text=f"'{config.TARGET_FOLDER}' 감시 시작...")
            
            # 4. 모니터링 스레드 시작
            self.monitor_thread = threading.Thread(target=self.monitor_folder, daemon=True)
            self.monitor_thread.start()

            # 5. GUI 업데이트 루프 시작
            self.root.after(100, self.update_plot)

        except FileNotFoundError as e:
            self.status_label.config(text=f"[오류] 필수 파일 없음: {e.filename}")
        except Exception as e:
            self.status_label.config(text=f"[오류] 초기화 실패: {e}")

    def load_and_stitch_files(self, file_paths):
        """
        폴더 내의 모든 replay.json 파일을 순서대로 연결하고 보간합니다.
        """
        all_waveforms = []
        all_timestamps = []
        
        for file_name in file_paths:
            try:
                with open(file_name, 'r') as f:
                    data = json.load(f)
                for frame in data.get('data', []):
                    if 'vitals' in frame.get('frameData', {}):
                        ts = frame.get('timestamp')
                        wf = frame['frameData']['vitals'].get('heartWaveform')
                        if ts is not None and wf is not None and len(wf) == config.INPUT_FEATURES:
                            all_timestamps.append(ts)
                            # [0.0, ...] 값은 '측정 실패'로 간주하고 [np.nan, ...]으로 변환
                            if np.sum(np.abs(wf)) == 0.0:
                                all_waveforms.append([np.nan] * config.INPUT_FEATURES)
                            else:
                                all_waveforms.append(wf)
            except Exception:
                pass # 파일이 쓰기 중일 수 있으므로 일단 통과

        if not all_waveforms:
            return None

        # [0,0,0] (NaN) 값을 선형 보간으로 채움
        df = pd.DataFrame(all_waveforms)
        df.interpolate(method='linear', axis=0, inplace=True, limit_direction='both')
        df.fillna(0, inplace=True) # 보간 후에도 남은 NaN은 0으로 강제 처리
        
        return df.values, all_timestamps

    def predict_hr(self, window_data):
        """
        (10, 15) 윈도우를 받아 HR을 예측합니다.
        [디버깅 2차] "원본 윈도우"와 "최종 예측값"을 그대로 출력
        """
        try:
            # ==========================================================
            # 🐞 디버깅 (1): 모델에 들어가는 '원본' 윈도우 값
            # window_data는 (10, 15) 크기입니다.
            
            print("[디버그-Raw] 모델에 입력된 (10, 15) 윈도우 (정규화 전):")
            
            # (10, 15) 배열을 그대로 문자열로 변환하여 출력
            # (터미널에 배열 전체가 찍힙니다)
            print(window_data)
            
            # ==========================================================

            # 1. 정규화
            normalized_window = (window_data - self.mean) / self.std
            
            # 2. 텐서 변환 (Batch 1 추가)
            tensor_input = torch.tensor(normalized_window, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 3. 예측
            with torch.no_grad():
                prediction = self.model(tensor_input)
            
            prediction_value = prediction.item()
            
            # ==========================================================
            # 🐞 디버깅 (2): 최종 예측값
            print(f"[디버그-Prediction] 예측된 HR 값: {prediction_value:.2f}")
            print("==================================================") # 구분선
            # ==========================================================
            
            return prediction_value-10
        
        except Exception as e:
            print(f"[예측 오류] {e}")
            return None

    def monitor_folder(self):
        """
        [별도 스레드] TARGET_FOLDER를 주기적으로 스캔하여 최신 윈도우를 큐에 넣습니다.
        """
        while True:
            try:
                # 1. 폴더 내 모든 replay.json 스캔 및 정렬
                file_pattern = os.path.join(config.TARGET_FOLDER, 'replay_*.json')
                all_files = sorted(glob.glob(file_pattern))
                
                if not all_files:
                    self.status_label.config(text=f"'{config.TARGET_FOLDER}'에서 replay 파일을 찾는 중...")
                    time.sleep(config.POLL_INTERVAL_MS / 1000.0)
                    continue

                # 2. 파일 연결 및 보간
                processed_waveforms, processed_timestamps = self.load_and_stitch_files(all_files)
                
                current_waveform_count = len(processed_waveforms)

                # 3. 신규 데이터 확인
                # (새로운 파형이 추가되었고, 윈도우 크기(10)를 넘었는지 확인)
                if current_waveform_count > self.last_waveform_count and current_waveform_count >= config.WINDOW_SIZE:
                    
                    self.last_waveform_count = current_waveform_count
                    
                    # 4. 마지막 윈도우 추출 (가장 최신 데이터 10개)
                    last_window = processed_waveforms[-config.WINDOW_SIZE:]
                    
                    # 5. 예측
                    prediction = self.predict_hr(last_window)
                    
                    if prediction is not None:
                        # GUI 스레드로 예측값 전송
                        self.data_queue.put(prediction)
                        self.status_label.config(text=f"예측 완료 (HR: {prediction:.1f}) | 총 파형: {current_waveform_count}개")
                
                elif current_waveform_count < config.WINDOW_SIZE:
                    self.status_label.config(text=f"데이터 수집 중... ({current_waveform_count}/{config.WINDOW_SIZE})")

            except Exception as e:
                print(f"[모니터링 스레드 오류] {e}")
                
            time.sleep(config.POLL_INTERVAL_MS / 1000.0)

    def update_plot(self):
        """
        [메인 GUI 스레드] 큐를 확인하고 그래프를 업데이트합니다.
        """
        try:
            # 큐에서 모든 새 데이터를 가져옴 (보통 1개)
            while not self.data_queue.empty():
                new_hr = self.data_queue.get_nowait()
                
                # 그래프 데이터 업데이트 (왼쪽으로 밀기)
                self.plot_data.pop(0)
                self.plot_data.append(new_hr)
                
                # 그래프 다시 그리기
                self.line.set_ydata(self.plot_data)
                self.ax.draw_artist(self.ax.patch)
                self.ax.draw_artist(self.line)
                self.canvas.blit(self.ax.bbox)
                self.canvas.flush_events()

        except queue.Empty:
            pass # 큐가 비어있으면 아무것도 안 함
        
        # 다음 업데이트 예약
        self.root.after(100, self.update_plot)

# --- 4. 애플리케이션 실행 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeHRApp(root)
    root.mainloop()