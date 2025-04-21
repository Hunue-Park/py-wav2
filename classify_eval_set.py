# 기존 파일 분류 스크립트 (분류 후 복사)
import shutil
from pathlib import Path

def organize_files():
    # 파일 분류를 위한 리스트 (파일명만 작성)
    correct_files = ["KsponSpeech_253001", "KsponSpeech_253002", "KsponSpeech_253003", "KsponSpeech_253004", "KsponSpeech_253005", "KsponSpeech_253006", "KsponSpeech_253007", "KsponSpeech_253008", "KsponSpeech_253009", "KsponSpeech_253010", ]  # 정확한 발음 파일 목록
    wrong_files = ["KsponSpeech_253011", "KsponSpeech_253012", "KsponSpeech_253013", "KsponSpeech_253014", "KsponSpeech_253015", "KsponSpeech_253016", "KsponSpeech_253017", "KsponSpeech_253018", "KsponSpeech_253019", "KsponSpeech_253020"]    # 틀린 발음 파일 목록
    
    source_dir = Path("./env/eval_set")
    correct_dir = source_dir / "correct"
    wrong_dir = source_dir / "wrong"
    
    # 디렉토리 생성
    correct_dir.mkdir(exist_ok=True)
    wrong_dir.mkdir(exist_ok=True)
    
    # 파일 복사
    for file_base in correct_files:
        for ext in [".pcm", ".txt"]:
            src = source_dir / f"{file_base}{ext}"
            if src.exists():
                shutil.copy(src, correct_dir)
    
    for file_base in wrong_files:
        for ext in [".pcm", ".txt"]:
            src = source_dir / f"{file_base}{ext}"
            if src.exists():
                shutil.copy(src, wrong_dir)
    
    print("파일 분류 완료")

if __name__ == "__main__":
    organize_files()