import os
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pyaudio
import wave
import numpy as np
from w2v_onnx_engine import Wav2VecCTCOnnxEngine

class RealtimeAudioRecorder:
    def __init__(self, output_dir, processor=None, callback_interval=1.0, filename_prefix="recorded_audio_"):
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.callback_interval = callback_interval
        self.processor = processor  # 프로세서 참조 추가
        self.is_recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.current_file = None
        self.recording_thread = None
        self.temp_file_thread = None
        self.stop_event = threading.Event()
        
    def start_recording(self):
        self.frames = []
        self.is_recording = True
        self.stop_event.clear()
        
        # 녹음 타임스탬프로 파일명 생성
        timestamp = int(time.time())
        self.current_file = os.path.join(
            self.output_dir, 
            f"{self.filename_prefix}{timestamp}.wav"
        )
        
        # 빈 파일 먼저 생성 (모니터링을 위해)
        wf = wave.open(self.current_file, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b'')  # 빈 데이터 쓰기
        wf.close()
        
        # 프로세서에 파일 추가 (여기로 이동)
        if self.processor:
            self.processor.add_file(self.current_file)
            print(f"모니터링 시작: {self.current_file}")
        
        # 오디오 스트림 시작
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self._callback
        )
        self.stream.start_stream()
        print("녹음 시작...")
        
        # 중간 파일 저장 스레드 시작
        self.temp_file_thread = threading.Thread(target=self._save_temp_files)
        self.temp_file_thread.daemon = True
        self.temp_file_thread.start()
        
    def _callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)
    
    def _save_temp_files(self):
        """실시간 처리를 위해 주기적으로 임시 파일 저장"""
        while not self.stop_event.is_set():
            if self.is_recording and self.frames:
                # 현재까지의 프레임으로 임시 파일 저장
                self._write_current_frames()
            time.sleep(self.callback_interval)
            
    def _write_current_frames(self):
        """현재까지 녹음된 프레임을 파일로 저장"""
        try:
            # 현재 프레임 복사 (스레드 안전성)
            current_frames = self.frames.copy()
            
            if not current_frames:
                return
                
            temp_file = f"{self.current_file}.temp"
            
            wf = wave.open(temp_file, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(current_frames))
            wf.close()
            
            # 완성된 임시 파일을 현재 파일로 이동 (원자적 연산)
            os.rename(temp_file, self.current_file)
        except Exception as e:
            print(f"임시 파일 저장 중 오류: {e}")
    
    def stop_recording(self):
        if not self.is_recording:
            return None
            
        self.is_recording = False
        self.stop_event.set()
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # 최종 파일 저장
        if self.frames:
            wf = wave.open(self.current_file, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(self.frames))
            wf.close()
        
        print(f"녹음 완료: {self.current_file}")
        final_path = self.current_file
        self.current_file = None
        return final_path
        
    def __del__(self):
        self.stop_event.set()
        if self.temp_file_thread and self.temp_file_thread.is_alive():
            self.temp_file_thread.join(timeout=1.0)
        self.audio.terminate()


class RealtimeAudioProcessor:
    def __init__(self, engine, reference_text, interval=0.5):
        self.engine = engine
        self.reference_text = reference_text
        self.interval = interval
        self.active_files = {}  # {파일경로: 마지막 수정 시간}
        self.stop_event = threading.Event()
        self.processing_thread = None
        self.min_file_size = 100  # 최소 파일 크기를 100바이트로 줄임
        
    def start_monitoring(self):
        self.stop_event.clear()
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop_monitoring(self):
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        self.active_files.clear()
    
    def add_file(self, file_path):
        """모니터링할 파일 추가"""
        if os.path.exists(file_path):
            self.active_files[file_path] = os.path.getmtime(file_path)
            print(f"파일 모니터링 시작: {file_path}")
    
    def remove_file(self, file_path):
        """모니터링에서 파일 제거"""
        if file_path in self.active_files:
            del self.active_files[file_path]
            print(f"파일 모니터링 중지: {file_path}")
            
    def _process_loop(self):
        """계속해서 파일들을 확인하고 변경된 경우 처리"""
        last_results = {}  # 이전 결과 저장
        
        while not self.stop_event.is_set():
            for file_path in list(self.active_files.keys()):
                try:
                    if not os.path.exists(file_path):
                        print(f"파일이 존재하지 않음: {file_path}")
                        continue
                        
                    # 파일이 변경되었는지 확인
                    current_mtime = os.path.getmtime(file_path)
                    current_size = os.path.getsize(file_path)
                    
                    if current_mtime > self.active_files[file_path]:
                        self.active_files[file_path] = current_mtime
                        
                        # 파일 크기가 너무 작으면 건너뛰기 (최소 크기 축소)
                        if current_size < self.min_file_size:
                            print(f"파일이 너무 작음 ({current_size} < {self.min_file_size})")
                            continue
                            
                        # 파일 처리
                        try:
                            result = self.engine.calculate_gop(file_path, self.reference_text)
                            
                            # 이전 결과와 비교하여 변경된 경우만 출력
                            if file_path not in last_results or result != last_results[file_path]:
                                last_results[file_path] = result
                                self._print_result(result)
                            else:
                                print("이전 결과와 동일, 출력 생략")
                                
                        except Exception as e:
                            print(f"파일 처리 중 오류 발생: {e}")
                except Exception as e:
                    print(f"파일 모니터링 중 오류: {e}")
                    
            time.sleep(self.interval)
    
    def _print_result(self, result):
        """결과 출력"""
        print(f"\n===== 실시간 발음 평가 결과 =====")
        print(f"전체 점수: {result['overall']}")
        
        print("\n단어별 점수:")
        for word_data in result["words"]:
            word = word_data["word"]
            score = word_data["scores"]["pronunciation"]
            print(f"- {word}: {score}")
        
        print("========================\n")


def main():
    # 설정
    MODEL_PATH = "./env/wav2vec2_ctc_dynamic.onnx"
    TOKENIZER_PATH = "./env/fine-tuned-wav2vec2-kspon/tokenizer.json"
    REFERENCE_TEXT = "참으로 위대한 일은 언제나 서서히 이루어지고 눈에 보이지 않게 성장해 가는 법이다."
    OUTPUT_DIR = "recordings"
    
    # 출력 디렉토리 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 엔진 초기화
    print("모델 로딩 중...")
    engine = Wav2VecCTCOnnxEngine(MODEL_PATH, TOKENIZER_PATH)
    print("모델 로딩 완료!")
    
    # 실시간 프로세서 생성
    processor = RealtimeAudioProcessor(engine, REFERENCE_TEXT, interval=0.1)
    processor.start_monitoring()
    
    # 녹음기 생성 (프로세서 참조 전달)
    recorder = RealtimeAudioRecorder(OUTPUT_DIR, processor=processor, callback_interval=1.0)
    
    try:
        print(f"\n정답 텍스트: {REFERENCE_TEXT}")
        print("\n명령어:")
        print("r: 녹음 시작/종료")
        print("q: 프로그램 종료")
        
        recording = False
        
        while True:
            cmd = input("\n> ")
            
            if cmd.lower() == 'r':
                if not recording:
                    # 녹음 시작
                    recorder.start_recording()
                    recording = True
                    # 프로세서에 파일 추가는 recorder 내부에서 처리
                else:
                    # 녹음 종료
                    final_path = recorder.stop_recording()
                    recording = False
                    # 모니터링에서 제거
                    processor.remove_file(final_path)
            elif cmd.lower() == 'q':
                if recording:
                    recorder.stop_recording()
                break
            else:
                print("알 수 없는 명령어입니다. 'r'로 녹음 시작/종료, 'q'로 종료하세요.")
    
    except KeyboardInterrupt:
        print("프로그램을 종료합니다...")
    finally:
        processor.stop_monitoring()


if __name__ == "__main__":
    main()