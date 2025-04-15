# check_encoding.py
import os
import sys

def try_read_file(file_path, encodings):
    """여러 인코딩으로 파일 읽기 시도"""
    results = {}
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read().strip()
                results[encoding] = content
                print(f"\n[{encoding}] 인코딩으로 성공적으로 읽음:")
                print("-" * 50)
                print(content)
                print("-" * 50)
        except UnicodeDecodeError:
            results[encoding] = None
            print(f"\n[{encoding}] 인코딩으로 읽기 실패")
    
    return results

def main():
    # 확인할 파일 경로
    folder_path = "env/KsponSpeech_03/KsponSpeech_0249"
    
    # 폴더 내 .txt 파일 찾기
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"'{folder_path}' 폴더에 .txt 파일이 없습니다.")
        return
    
    # 첫 번째 txt 파일 선택
    sample_file = txt_files[0]
    for sample_file in txt_files:
        file_path = os.path.join(folder_path, sample_file)
        
        print(f"파일 '{file_path}' 확인 중...\n")
        
        # 여러 인코딩으로 시도
        encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1', 'ascii']
        results = try_read_file(file_path, encodings)
        
        # 성공한 인코딩 확인
        successful = [enc for enc, content in results.items() if content is not None]
    
        print("\n결과 요약:")
        if successful:
            print(f"파일을 성공적으로 읽을 수 있는 인코딩: {', '.join(successful)}")
            print(f"추천 인코딩: {successful[0]}")
        else:
            print("어떤 인코딩으로도 파일을 읽을 수 없습니다.")

if __name__ == "__main__":
    main()