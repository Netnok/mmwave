import os
import glob
import json
import shutil
import time
import re
import sys

# --- CONFIGURATION (설정) ---
# 이 설정은 config.py 파일의 설정과 동일하게 맞춰야 합니다.
# 실제 운영 시에는 config.py에서 직접 읽어오는 로직을 추가하는 것이 좋으나,
# 지금은 독립적인 실행을 위해 직접 정의합니다.

# 센서가 데이터를 쓰는 원본 폴더 (GUI는 접근 금지)
SOURCE_FOLDER = r'dataset\H4\15'
# GUI가 안전하게 읽을 섀도우 폴더 (Copier만 쓰기 가능)
SHADOW_FOLDER = 'autologger'
# 파일 감시 주기 (초)
POLL_INTERVAL_SEC = 1.0


def setup_folders():
    """필요한 폴더를 생성합니다."""
    os.makedirs(SOURCE_FOLDER, exist_ok=True)
    os.makedirs(SHADOW_FOLDER, exist_ok=True)
    print(f"[Setup] 원본 폴더 (쓰기): {SOURCE_FOLDER}")
    print(f"[Setup] 섀도우 폴더 (읽기): {SHADOW_FOLDER}")


def get_sorted_files(folder):
    """폴더 내 replay_*.json 파일을 숫자 순으로 정렬하여 반환합니다."""
    file_pattern = os.path.join(folder, 'replay_*.json')
    all_files_unsorted = glob.glob(file_pattern)

    def get_file_number(file_path):
        match = re.search(r'replay_(\d+)\.json', os.path.basename(file_path))
        if match:
            return int(match.group(1))
        return -1

    return sorted(all_files_unsorted, key=get_file_number)


def copy_completed_files():
    """원본 폴더를 감시하고, Lock이 풀린 파일을 섀도우 폴더로 복사합니다."""
    source_files = get_sorted_files(SOURCE_FOLDER)
    
    if not source_files:
        print("[Monitor] 원본 폴더에 파일 없음. 대기 중...")
        return
    
    # Lock이 풀린 (쓰기가 완료된) 파일만 복사
    copied_count = 0
    for src_file in source_files:
        filename = os.path.basename(src_file)
        dest_file = os.path.join(SHADOW_FOLDER, filename)
        
        try:
            # 1. 파일 열기 시도 (Lock 확인)
            # 파일을 '읽기/쓰기 모드'로 열어 Lock이 걸려있는지 확인합니다.
            # Lock이 걸려있다면 여기서 에러가 발생합니다.
            with open(src_file, 'r+'):
                 pass

            # 2. Lock이 없다면 -> 복사 시도
            if not os.path.exists(dest_file) or os.path.getsize(src_file) != os.path.getsize(dest_file):
                # 파일이 없거나 크기가 다르다면 (업데이트 필요) 복사
                shutil.copy2(src_file, dest_file) # 메타데이터도 함께 복사
                print(f"[Copy OK] {filename} 복사 완료.")
                copied_count += 1
            
        except (IOError, PermissionError):
            # Lock 걸림 (쓰기 중) 또는 OS 권한 오류
            print(f"[Wait] {filename}은 현재 쓰기 중이거나 잠겨있음. 건너뜀.")
            
        except Exception as e:
            print(f"[Error] {filename} 처리 중 예상치 못한 오류 발생: {e}")
            
    if copied_count == 0:
        print("[Monitor] 복사할 새 파일(완료된) 없음. 대기 중...")


def main():
    setup_folders()
    print("--- Auto Copier 실행 중 (Ctrl+C로 종료) ---")
    while True:
        try:
            copy_completed_files()
            time.sleep(POLL_INTERVAL_SEC)
        except KeyboardInterrupt:
            print("\nAuto Copier를 종료합니다.")
            sys.exit(0)
        except Exception as e:
            print(f"\n[Fatal Error] {e}. 5초 후 재시도.")
            time.sleep(5)

if __name__ == '__main__':
    main()