# prepare_kspon_test.py
import os
import shutil
import random

def prepare_kspon_test(transcript_file, pcm_source_dir, target_dir, sample_count=30):
    """KsponSpeech eval_clean에서 평가용 샘플 추출"""
    os.makedirs(target_dir, exist_ok=True)
    
    # 트랜스크립트 파일 로드
    samples = []
    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' :: ')
            if len(parts) == 2:
                pcm_path, text = parts
                file_id = os.path.basename(pcm_path).replace('.pcm', '')
                samples.append((pcm_path, file_id, text))
    
    # 샘플 추출
    selected_samples = random.sample(samples, min(sample_count, len(samples)))
    
    # 파일 복사 및 텍스트 저장
    for pcm_path, file_id, text in selected_samples:
        # PCM 파일 경로
        source_path = os.path.join(pcm_source_dir, pcm_path)
        
        if os.path.exists(source_path):
            # PCM 파일 복사
            target_pcm = os.path.join(target_dir, f"{file_id}.pcm")
            shutil.copy2(source_path, target_pcm)
            
            # 텍스트 전처리 및 저장
            clean_text = preprocess_ksponspeech_text(text)
            with open(os.path.join(target_dir, f"{file_id}.txt"), 'w', encoding='utf-8') as f:
                f.write(clean_text)
            
            print(f"복사 완료: {file_id}")
        else:
            print(f"파일 없음: {source_path}")
    
    print(f"총 {len(selected_samples)}개 샘플 추출 완료: {target_dir}")

def preprocess_ksponspeech_text(text):
    """KsponSpeech 텍스트 전처리"""
    # 잡음 레이블 제거 (o/, b/, +, * 등)
    import re
    text = re.sub(r'[/][a-z+*]+[/]', '', text)
    
    # 숫자 표현 정규화 - (40분)/(사십 분) 형태에서 '사십 분' 선택
    numbers = re.findall(r'\([^)]*\)/\([^)]*\)', text)
    for num in numbers:
        parts = num.split('/')
        if len(parts) == 2:
            # 한글 표현 선택 (두번째 괄호)
            korean_num = parts[1].strip('()')
            text = text.replace(num, korean_num)
    
    # 남은 괄호 제거
    text = re.sub(r'\([^)]*\)', '', text)
    
    # 특수문자 제거
    text = re.sub(r'[^\w\s]', '', text)
    
    # 여러 공백 정규화
    text = ' '.join(text.split())
    
    return text

# 실행
prepare_kspon_test(
    transcript_file="env/kspon_scripts/eval_clean.trn",
    pcm_source_dir="env",  # PCM 파일 기준 경로
    target_dir="./kspon_test_samples",
    sample_count=30
)