# finetune_wav2vec2.py
import os
import sys

# MPS 관련 환경 변수를 더 명확하게 비활성화
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_DISABLE"] = "1"
os.environ["USE_MPS"] = "0"

import torch
# PyTorch MPS 백엔드 명시적으로 비활성화
# Python 시작부터 MPS 감지를 무시하도록 합니다
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
    torch.backends.mps.enabled = False

import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
)
from data_processing import load_pcm_file
import warnings
import argparse

warnings.filterwarnings("ignore")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PyTorch 장치 설정 강제
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CUDA 비활성화
torch.set_default_device("cpu")          # 기본 장치를 CPU로 설정

# 1. 데이터셋 클래스 정의
class KsponDataset(Dataset):
    def __init__(self, data_dir, processor):
        """
        KsponSpeech_03 데이터셋을 로드합니다.
        
        Args:
            data_dir: KsponSpeech_03 디렉토리 경로
            processor: 모델 프로세서
        """
        self.processor = processor
        self.data_dir = data_dir
        
        # 오디오-텍스트 쌍 로드
        self.pairs = []
        self._load_pairs()
        logger.info(f"총 {len(self.pairs)}개의 오디오-텍스트 쌍 로드 완료")
        
    def _load_pairs(self):
        """데이터 디렉토리에서 오디오-텍스트 쌍 찾기"""
        # 하위 디렉토리 필터링 (KsponSpeech_0249만 사용)
        target_subdir = "KsponSpeech_0249"  # 특정 하위 디렉토리만 사용
        
        subdir_path = os.path.join(self.data_dir, target_subdir)
        if os.path.isdir(subdir_path):
            logger.info(f"하위 디렉토리 '{target_subdir}' 탐색 중...")
            
            # 해당 하위 디렉토리에서 PCM 파일 찾기
            pcm_files = [f for f in os.listdir(subdir_path) if f.endswith('.pcm')]
            
            for pcm_file in pcm_files:
                base_name = os.path.splitext(pcm_file)[0]
                txt_file = f"{base_name}.txt"
                pcm_path = os.path.join(subdir_path, pcm_file)
                txt_path = os.path.join(subdir_path, txt_file)
                
                # 텍스트 파일이 있는 경우만 쌍으로 추가
                if os.path.exists(txt_path):
                    try:
                        # 단순하게 euc-kr 인코딩으로만 시도
                        with open(txt_path, 'r', encoding='euc-kr') as f:
                            text = f.read().strip()
                        
                        # 유효한 텍스트가 있는 경우만 추가 
                        if text:
                            # 디버깅용 로그 추가
                            logger.info(f"파일 '{txt_path}' 로드 완료: 텍스트 길이 {len(text)}")
                            
                            # 오디오-텍스트 쌍 추가
                            self.pairs.append({
                                "audio_path": pcm_path,
                                "text": text
                            })
                    except UnicodeDecodeError as e:
                        logger.warning(f"텍스트 파일 '{txt_path}' 인코딩 오류: {str(e)}")
                    except Exception as e:
                        logger.error(f"텍스트 파일 '{txt_path}' 처리 중 오류 발생: {str(e)}")
                        logger.error(f"오류 유형: {type(e).__name__}")
                        import traceback
                        logger.error(f"스택 트레이스: {traceback.format_exc()}")
        else:
            logger.error(f"지정된 하위 디렉토리 '{target_subdir}'를 찾을 수 없습니다.")
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        audio_path = pair["audio_path"]
        text = pair["text"]
        
        try:
            # PCM 파일 로드
            audio = load_pcm_file(audio_path)
            
            # 텍스트 전처리 (라벨 토큰화)
            try:
                with self.processor.as_target_processor():
                    labels = self.processor(text).input_ids
            except Exception as e:
                logger.error(f"텍스트 '{text[:50]}...' 토큰화 중 오류: {str(e)}")
                # 임시 해결책: 텍스트에 특수 문자가 있으면 제거
                text = ''.join(char for char in text if char.isalnum() or char.isspace())
                with self.processor.as_target_processor():
                    labels = self.processor(text or " ").input_ids
                
            # 오디오 전처리
            input_values = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors='pt'
            ).input_values.squeeze()
                
            return {
                "input_values": input_values,
                "labels": torch.tensor(labels)
            }
        except Exception as e:
            logger.error(f"데이터 로드 실패 {audio_path}: {str(e)}")
            logger.error(f"오류 유형: {type(e).__name__}")
            
            # 실패 시 다른 인덱스 시도
            if len(self) > 1:
                alt_idx = (idx + 1) % len(self)
                logger.info(f"대체 데이터 {alt_idx} 시도")
                return self.__getitem__(alt_idx)
            else:
                # 대체 데이터가 없는 경우 빈 데이터 반환
                logger.error("대체 데이터 없음, 더미 데이터 반환")
                # 최소한의 더미 데이터
                return {
                    "input_values": torch.zeros(16000),
                    "labels": torch.tensor([0])
                }

# 2. 데이터 콜레이션 함수를 클래스로 변경
class DataCollatorCTCWithPadding:
    def __init__(self, processor):
        self.processor = processor
        
    def __call__(self, batch):
        """배치 데이터를 패딩하여 처리"""
        input_features = [{"input_values": item["input_values"]} for item in batch]
        label_features = [{"input_ids": item["labels"]} for item in batch]
        
        # 입력 패딩
        batch = self.processor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )
        
        # 라벨 패딩
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=True,
                return_tensors="pt",
            )
        
        # 라벨 처리
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        batch["labels"] = labels
        
        return batch

# 3. 학습 진행률 추적 및 체크포인트 관리를 위한 콜백
class SaveProgressCallback(TrainerCallback):
    """학습 진행 상황을 저장하고 체크포인트를 관리하는 콜백"""
    def on_save(self, args, state, control, **kwargs):
        # 체크포인트 저장 시 호출
        with open(os.path.join(args.output_dir, "last_checkpoint.txt"), "w") as f:
            f.write(f"checkpoint-{state.global_step}")
        return control

# 4. 메인 함수
def main(args):
    """wav2vec2 모델 파인튜닝"""
    # 기본 장치 강제로 CPU로 설정
    torch._C._set_backcompat_keepdim_warn(False)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        # MPS 비활성화
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.backends.mps.enabled = False
    
    # 디버그용 장치 정보 출력
    logger.info(f"PyTorch 사용 장치: {device}")
    logger.info(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if hasattr(torch.backends, 'mps'):
        logger.info(f"MPS 사용 가능: {torch.backends.mps.is_available()}")
        logger.info(f"MPS 활성화됨: {torch.backends.mps.enabled}")
    
    global processor  # data_collator에서 접근
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 모델 및 프로세서 로드
    logger.info(f"모델 '{args.model_name}' 로드 중...")
    
    # 캐시 디렉토리 설정
    cache_dir = Path.home() / ".cache" / "py-wav2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 체크포인트에서 이어서 학습하는 경우
    if args.resume_from_checkpoint:
        if os.path.isdir(args.resume_from_checkpoint):
            checkpoint_path = args.resume_from_checkpoint
        else:
            # checkpoint-1234 형식인 경우
            checkpoint_path = os.path.join(args.output_dir, args.resume_from_checkpoint)
            if not os.path.isdir(checkpoint_path) and args.resume_from_checkpoint.startswith("checkpoint-"):
                # 숫자만 있는 경우 (예: 1234)
                step_num = args.resume_from_checkpoint.split("-")[-1]
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{step_num}")
        
        logger.info(f"체크포인트 '{checkpoint_path}'에서 학습 재개")
        processor = Wav2Vec2Processor.from_pretrained(checkpoint_path)
        model = Wav2Vec2ForCTC.from_pretrained(checkpoint_path)
        # CPU 모드로 강제 설정
        model = model.to('cpu')
    else:
        # 새로 학습 시작
        processor = Wav2Vec2Processor.from_pretrained(args.model_name, cache_dir=cache_dir)
        model = Wav2Vec2ForCTC.from_pretrained(args.model_name, cache_dir=cache_dir)
        # CPU 모드로 강제 설정
        model = model.to('cpu')
    
    # 매우 강력하게 CPU로 이동 강제
    for param in model.parameters():
        param.data = param.data.to("cpu")
    model = model.to("cpu")
    
    # 2. 데이터셋 로드
    logger.info(f"KsponSpeech 데이터셋 '{args.data_dir}' 로드 중...")
    
    # 데이터셋 생성
    dataset = KsponDataset(
        data_dir=args.data_dir,
        processor=processor
    )
    
    if len(dataset) == 0:
        logger.error(f"데이터셋에 유효한 오디오-텍스트 쌍이 없습니다. 경로 확인: {args.data_dir}")
        return
    
    # 3. 훈련 설정
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # evaluation_strategy="no",  # 평가 없음
        num_train_epochs=args.epochs,
        fp16=False,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,  # 전체 스텝의 10%를 워밍업에 사용
        save_total_limit=3,
        push_to_hub=False,
        dataloader_num_workers=args.num_workers,
        disable_tqdm=args.disable_tqdm,
        # load_best_model_at_end=False,
        # 추가 옵션
        gradient_checkpointing=False,  # 메모리 효율성 개선
        logging_dir=os.path.join(args.output_dir, "logs"),
        use_cpu=True,  # CPU 사용 강제
        no_cuda=True,  # CUDA 비활성화
    )
    
    # 4. 트레이너 초기화
    data_collator = DataCollatorCTCWithPadding(processor)
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor.feature_extractor,
        callbacks=[SaveProgressCallback],
    )
    
    # 5. 파인튜닝 시작
    logger.info("파인튜닝 시작...")
    
    # 체크포인트에서 이어서 학습하는 경우
    checkpoint_path = None
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
    else:
        # 자동으로 마지막 체크포인트 탐색
        last_checkpoint_file = os.path.join(args.output_dir, "last_checkpoint.txt")
        if os.path.exists(last_checkpoint_file):
            with open(last_checkpoint_file, "r") as f:
                last_checkpoint = f.read().strip()
                checkpoint_path = os.path.join(args.output_dir, last_checkpoint)
                if os.path.exists(checkpoint_path):
                    logger.info(f"마지막 체크포인트 '{checkpoint_path}'에서 이어서 학습합니다.")
    
    trainer.train(resume_from_checkpoint=checkpoint_path)
    
    # 6. 모델 저장
    logger.info(f"파인튜닝된 모델 저장: {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    logger.info("파인튜닝 완료!")

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="Wav2Vec2 모델 파인튜닝")
    parser.add_argument("--model_name", type=str, default="kresnik/wav2vec2-large-xlsr-korean",
                        help="파인튜닝할 사전학습 모델 이름")
    parser.add_argument("--data_dir", type=str, default="env/KsponSpeech_03",
                        help="KsponSpeech_03 디렉토리 경로")
    parser.add_argument("--output_dir", type=str, default="fine-tuned-wav2vec2-kspon",
                        help="파인튜닝된 모델 저장 경로")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="배치 크기")
    parser.add_argument("--epochs", type=int, default=5,
                        help="훈련 에포크 수")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="학습률")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="모델 저장 주기")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="로깅 주기")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="그래디언트 누적 스텝 수")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="데이터 로더 워커 수")
    parser.add_argument("--disable_tqdm", action="store_true",
                        help="진행 표시줄 비활성화 (서버 환경)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="그래디언트 체크포인팅 사용 (메모리 절약)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="체크포인트에서 이어서 학습 (디렉토리 경로)")
    
    args = parser.parse_args()
    
    main(args)