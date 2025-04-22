# evaluate_gop_cases.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# from w2v_onnx_engine import Wav2VecCTCOnnxEngine
import soundfile as sf
from tqdm import tqdm
import wave
import time
from wav2vec_cpp_py import Wav2VecCTCOnnxEngine

# 기존 gop_system_eval.py의 함수 재사용
def read_pcm_file(file_path, sample_width=2, channels=1, sample_rate=16000):
    """PCM 파일을 WAV로 변환하여 읽기"""
    temp_wav = file_path.with_suffix('.temp.wav')
    
    with open(file_path, 'rb') as pcm_file:
        pcm_data = pcm_file.read()
    
    with wave.open(str(temp_wav), 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    
    audio, sr = sf.read(temp_wav)
    
    if temp_wav.exists():
        temp_wav.unlink()
    
    return audio, sr

def read_text_with_encoding(file_path):
    """여러 인코딩을 시도하여 텍스트 파일 읽기"""
    encodings = ['cp949', 'euc-kr', 'utf-8']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read().strip()
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"지원되는 모든 인코딩으로 파일을 읽을 수 없습니다: {file_path}")

def evaluate_cases():
    # 경로 설정
    base_dir = Path("./env/eval_set")
    correct_dir = base_dir / "correct"  # 정확한 발음 케이스
    wrong_dir = base_dir / "wrong"      # 틀린 발음 케이스
    output_dir = Path("./results/case_study")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 디렉토리 생성
    correct_dir.mkdir(exist_ok=True)
    wrong_dir.mkdir(exist_ok=True)
    
    # ONNX 엔진 초기화
    model_path = "./env/wav2vec2_ctc_dynamic.onnx"
    tokenizer_path = "./env/fine-tuned-wav2vec2-kspon/tokenizer.json"
    
    engine = Wav2VecCTCOnnxEngine(model_path, tokenizer_path)
    print("엔진 초기화 완료")
    
    # 결과 저장 리스트
    correct_results = []
    wrong_results = []
    
    # 실행 시간 저장 리스트
    correct_times = []
    wrong_times = []
    
    # 평가 함수
    def evaluate_dir(dir_path, results_list, times_list, category):
        pcm_files = sorted(dir_path.glob("*.pcm"))
        
        for pcm_file in tqdm(pcm_files, desc=f"{category} 케이스 평가 중"):
            txt_file = pcm_file.with_suffix('.txt')
            
            if not txt_file.exists():
                print(f"텍스트 파일 없음: {txt_file}")
                continue
            
            try:
                text = read_text_with_encoding(txt_file)
                temp_wav = pcm_file.with_suffix('.temp.wav')
                
                # PCM 파일 처리
                audio, sr = read_pcm_file(pcm_file)
                sf.write(temp_wav, audio, sr)
                
                # GOP 계산 시간 측정 시작
                start_time = time.time()
                gop_result = engine.calculate_gop(str(temp_wav), text)
                end_time = time.time()
                execution_time = end_time - start_time
                times_list.append(execution_time)
                
                # 결과 저장
                result = {
                    "file": pcm_file.name,
                    "text": text,
                    "gop": gop_result,
                    "category": category,
                    "execution_time": execution_time
                }
                results_list.append(result)
                
                # 개별 결과 저장
                # with open(output_dir / f"{category}_{pcm_file.stem}_result.json", 'w', encoding='utf-8') as f:
                #     json.dump(result, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                print(f"처리 실패: {pcm_file} - {e}")
            finally:
                if temp_wav.exists():
                    temp_wav.unlink()
    
    # 두 디렉토리 평가
    evaluate_dir(correct_dir, correct_results, correct_times, "correct")
    evaluate_dir(wrong_dir, wrong_results, wrong_times, "wrong")
    
    # 실행 시간 통계
    all_times = correct_times + wrong_times
    
    if all_times:
        avg_time = np.mean(all_times)
        correct_avg_time = np.mean(correct_times) if correct_times else 0
        wrong_avg_time = np.mean(wrong_times) if wrong_times else 0
        
        time_stats = {
            "overall_average_time": float(avg_time),
            "correct_cases_average_time": float(correct_avg_time),
            "wrong_cases_average_time": float(wrong_avg_time),
            "total_files_processed": len(all_times),
            "min_time": float(min(all_times)),
            "max_time": float(max(all_times))
        }
        
        # 시간 통계 저장
        with open(output_dir / "execution_time_stats.json", 'w', encoding='utf-8') as f:
            json.dump(time_stats, f, ensure_ascii=False, indent=2)
    
    # 결과 분석
    if correct_results and wrong_results:
        correct_scores = [r["gop"]["overall"] for r in correct_results]
        wrong_scores = [r["gop"]["overall"] for r in wrong_results]
        
        correct_avg = np.mean(correct_scores)
        wrong_avg = np.mean(wrong_scores)
        
        # 결과 요약
        summary = {
            "correct_cases": {
                "count": len(correct_results),
                "average_score": float(correct_avg),
                "min_score": float(min(correct_scores)),
                "max_score": float(max(correct_scores)),
                "average_execution_time": float(correct_avg_time) if correct_times else 0
            },
            "wrong_cases": {
                "count": len(wrong_results),
                "average_score": float(wrong_avg),
                "min_score": float(min(wrong_scores)),
                "max_score": float(max(wrong_scores)),
                "average_execution_time": float(wrong_avg_time) if wrong_times else 0
            },
            "score_difference": float(correct_avg - wrong_avg),
            "overall_average_execution_time": float(avg_time) if all_times else 0
        }
        
        # 요약 저장
        with open(output_dir / "case_comparison.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 결과 시각화
        plt.figure(figsize=(10, 6))
        plt.boxplot([correct_scores, wrong_scores], labels=['정확한 발음', '틀린 발음'])
        plt.title('GOP 점수 분포 비교')
        plt.ylabel('GOP 점수')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(output_dir / "score_comparison.png")
        
        print(f"평가 완료:")
        print(f"- 정확한 발음 케이스: {len(correct_results)}개, 평균 점수: {correct_avg:.2f}")
        print(f"- 틀린 발음 케이스: {len(wrong_results)}개, 평균 점수: {wrong_avg:.2f}")
        print(f"- 점수 차이: {correct_avg - wrong_avg:.2f}")
        print(f"실행 시간 통계:")
        print(f"- 전체 평균 실행 시간: {avg_time:.4f}초")
        print(f"- 정확한 발음 케이스 평균 실행 시간: {correct_avg_time:.4f}초")
        print(f"- 틀린 발음 케이스 평균 실행 시간: {wrong_avg_time:.4f}초")
        print(f"- 최소 실행 시간: {min(all_times):.4f}초, 최대 실행 시간: {max(all_times):.4f}초")

if __name__ == "__main__":
    evaluate_cases()