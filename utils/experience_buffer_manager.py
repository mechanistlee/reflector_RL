"""
경험 버퍼 관리 유틸리티
====================

강화학습 경험 데이터를 효율적으로 저장/로드하는 기능 제공

주요 기능:
- HDF5, Pickle, NumPy 형식으로 경험 데이터 저장
- 데이터 압축 및 메타데이터 관리
- 오래된 파일 자동 정리
- 배치 로딩 및 메모리 효율적 처리
"""

import os
import time
import pickle
import numpy as np
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import logging
import glob


class ExperienceBufferManager:
    """경험 버퍼 데이터 저장/로드 관리자"""
    
    def __init__(self, 
                 save_path: str = "data/experience_buffer",
                 format_type: str = "hdf5",
                 max_files: int = 50,
                 compress: bool = True):
        """
        Args:
            save_path: 저장 디렉토리 경로
            format_type: 저장 형식 ("hdf5", "pickle", "numpy")
            max_files: 최대 보관 파일 개수
            compress: 압축 여부
        """
        self.save_path = save_path
        self.format_type = format_type.lower()
        self.max_files = max_files
        self.compress = compress
        
        # 디렉토리 생성
        os.makedirs(save_path, exist_ok=True)
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        
        # 지원하는 형식 검증
        if self.format_type not in ["hdf5", "pickle", "numpy"]:
            raise ValueError(f"지원하지 않는 형식: {format_type}. 지원 형식: hdf5, pickle, numpy")
        
        # HDF5 형식 사용 시 h5py 임포트 시도
        if self.format_type == "hdf5":
            try:
                import h5py
                self.h5py = h5py
            except ImportError:
                self.logger.warning("h5py가 설치되지 않아 pickle 형식으로 변경합니다.")
                self.format_type = "pickle"
    
    def save_experience_buffer(self, 
                              experiences: List[Dict[str, Any]], 
                              step_number: int,
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        경험 버퍼를 파일로 저장
        
        Args:
            experiences: 경험 데이터 리스트
            step_number: 현재 훈련 스텝 번호
            metadata: 추가 메타데이터
            
        Returns:
            저장된 파일 경로
        """
        if not experiences:
            self.logger.warning("저장할 경험 데이터가 없습니다.")
            return ""
        
        # 파일명 생성 (타임스탬프 + 스텝 번호)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experience_buffer_step{step_number:06d}_{timestamp}"
        
        # 메타데이터 준비
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "step_number": step_number,
            "timestamp": timestamp,
            "num_experiences": len(experiences),
            "format_type": self.format_type,
            "compressed": self.compress,
            "saved_at": time.time()
        })
        
        try:
            if self.format_type == "hdf5":
                filepath = self._save_hdf5(experiences, filename, metadata)
            elif self.format_type == "pickle":
                filepath = self._save_pickle(experiences, filename, metadata)
            elif self.format_type == "numpy":
                filepath = self._save_numpy(experiences, filename, metadata)
            else:
                raise ValueError(f"지원하지 않는 형식: {self.format_type}")
            
            self.logger.info(f"✅ 경험 버퍼 저장 완료: {filepath}")
            self.logger.info(f"   - 경험 개수: {len(experiences):,}")
            self.logger.info(f"   - 파일 크기: {self._get_file_size(filepath)}")
            
            # 오래된 파일 정리
            self._cleanup_old_files()
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 경험 버퍼 저장 실패: {e}")
            return ""
    
    def load_experience_buffer(self, 
                              filepath: str = None,
                              step_number: int = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        경험 버퍼를 파일에서 로드
        
        Args:
            filepath: 로드할 파일 경로 (None이면 가장 최근 파일)
            step_number: 특정 스텝의 파일 로드
            
        Returns:
            (경험 데이터 리스트, 메타데이터)
        """
        try:
            # 파일 경로 결정
            if filepath is None:
                if step_number is not None:
                    filepath = self._find_file_by_step(step_number)
                else:
                    filepath = self._get_latest_file()
            
            if not filepath or not os.path.exists(filepath):
                self.logger.warning(f"경험 버퍼 파일을 찾을 수 없습니다: {filepath}")
                return [], {}
            
            # 파일 형식 감지
            format_type = self._detect_file_format(filepath)
            
            if format_type == "hdf5":
                return self._load_hdf5(filepath)
            elif format_type == "pickle":
                return self._load_pickle(filepath)
            elif format_type == "numpy":
                return self._load_numpy(filepath)
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {format_type}")
                
        except Exception as e:
            self.logger.error(f"❌ 경험 버퍼 로드 실패: {e}")
            return [], {}
    
    def list_experience_files(self) -> List[Dict[str, Any]]:
        """저장된 경험 파일 목록 반환"""
        files_info = []
        
        patterns = [
            os.path.join(self.save_path, "experience_buffer_*.h5"),
            os.path.join(self.save_path, "experience_buffer_*.pkl"),
            os.path.join(self.save_path, "experience_buffer_*.npz")
        ]
        
        for pattern in patterns:
            for filepath in glob.glob(pattern):
                try:
                    file_stat = os.stat(filepath)
                    file_info = {
                        "filepath": filepath,
                        "filename": os.path.basename(filepath),
                        "size": file_stat.st_size,
                        "size_mb": file_stat.st_size / (1024 * 1024),
                        "created_time": file_stat.st_ctime,
                        "modified_time": file_stat.st_mtime,
                        "format": self._detect_file_format(filepath)
                    }
                    
                    # 스텝 번호 추출 시도
                    try:
                        step_str = filepath.split("step")[1].split("_")[0]
                        file_info["step_number"] = int(step_str)
                    except:
                        file_info["step_number"] = -1
                    
                    files_info.append(file_info)
                    
                except Exception as e:
                    self.logger.warning(f"파일 정보 읽기 실패 {filepath}: {e}")
        
        # 스텝 번호순으로 정렬
        files_info.sort(key=lambda x: x["step_number"])
        return files_info
    
    def _save_hdf5(self, experiences: List[Dict], filename: str, metadata: Dict) -> str:
        """HDF5 형식으로 저장"""
        filepath = os.path.join(self.save_path, f"{filename}.h5")
        
        with self.h5py.File(filepath, 'w') as f:
            # 메타데이터 저장
            meta_group = f.create_group("metadata")
            for key, value in metadata.items():
                if isinstance(value, (str, int, float)):
                    meta_group.attrs[key] = value
                else:
                    meta_group.attrs[key] = str(value)
            
            # 경험 데이터 저장
            exp_group = f.create_group("experiences")
            
            for i, experience in enumerate(experiences):
                exp_subgroup = exp_group.create_group(f"experience_{i}")
                
                for key, value in experience.items():
                    if isinstance(value, np.ndarray):
                        dataset = exp_subgroup.create_dataset(
                            key, data=value,
                            compression='gzip' if self.compress else None
                        )
                    elif isinstance(value, (int, float)):
                        exp_subgroup.attrs[key] = value
                    else:
                        exp_subgroup.attrs[key] = str(value)
        
        return filepath
    
    def _save_pickle(self, experiences: List[Dict], filename: str, metadata: Dict) -> str:
        """Pickle 형식으로 저장"""
        filepath = os.path.join(self.save_path, f"{filename}.pkl")
        
        data = {
            "metadata": metadata,
            "experiences": experiences
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return filepath
    
    def _save_numpy(self, experiences: List[Dict], filename: str, metadata: Dict) -> str:
        """NumPy 형식으로 저장"""
        filepath = os.path.join(self.save_path, f"{filename}.npz")
        
        # 경험 데이터를 numpy 배열로 변환
        arrays_dict = {"metadata": json.dumps(metadata)}
        
        for i, experience in enumerate(experiences):
            for key, value in experience.items():
                array_key = f"exp_{i}_{key}"
                if isinstance(value, np.ndarray):
                    arrays_dict[array_key] = value
                elif isinstance(value, (int, float)):
                    arrays_dict[array_key] = np.array([value])
                else:
                    arrays_dict[array_key] = np.array([str(value)])
        
        if self.compress:
            np.savez_compressed(filepath, **arrays_dict)
        else:
            np.savez(filepath, **arrays_dict)
        
        return filepath
    
    def _load_hdf5(self, filepath: str) -> Tuple[List[Dict], Dict]:
        """HDF5 형식에서 로드"""
        with self.h5py.File(filepath, 'r') as f:
            # 메타데이터 로드
            metadata = dict(f["metadata"].attrs)
            
            # 경험 데이터 로드
            experiences = []
            exp_group = f["experiences"]
            
            for exp_key in exp_group.keys():
                exp_subgroup = exp_group[exp_key]
                experience = {}
                
                # 데이터셋 로드
                for dataset_key in exp_subgroup.keys():
                    experience[dataset_key] = exp_subgroup[dataset_key][...]
                
                # 속성 로드
                for attr_key in exp_subgroup.attrs.keys():
                    experience[attr_key] = exp_subgroup.attrs[attr_key]
                
                experiences.append(experience)
        
        return experiences, metadata
    
    def _load_pickle(self, filepath: str) -> Tuple[List[Dict], Dict]:
        """Pickle 형식에서 로드"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return data["experiences"], data["metadata"]
    
    def _load_numpy(self, filepath: str) -> Tuple[List[Dict], Dict]:
        """NumPy 형식에서 로드"""
        data = np.load(filepath)
        
        # 메타데이터 로드
        metadata = json.loads(str(data["metadata"]))
        
        # 경험 데이터 복원
        experiences = []
        exp_dict = {}
        
        for key in data.keys():
            if key.startswith("exp_"):
                parts = key.split("_", 2)  # "exp", index, field_name
                exp_idx = int(parts[1])
                field_name = parts[2]
                
                if exp_idx not in exp_dict:
                    exp_dict[exp_idx] = {}
                
                value = data[key]
                if value.shape == (1,):
                    exp_dict[exp_idx][field_name] = value[0]
                else:
                    exp_dict[exp_idx][field_name] = value
        
        # 인덱스 순서로 정렬하여 리스트 생성
        for i in sorted(exp_dict.keys()):
            experiences.append(exp_dict[i])
        
        return experiences, metadata
    
    def _detect_file_format(self, filepath: str) -> str:
        """파일 확장자로 형식 감지"""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".h5":
            return "hdf5"
        elif ext == ".pkl":
            return "pickle"
        elif ext == ".npz":
            return "numpy"
        else:
            return "unknown"
    
    def _get_latest_file(self) -> str:
        """가장 최근 파일 경로 반환"""
        files = self.list_experience_files()
        if not files:
            return ""
        
        # 수정 시간 기준으로 가장 최근 파일 반환
        latest_file = max(files, key=lambda x: x["modified_time"])
        return latest_file["filepath"]
    
    def _find_file_by_step(self, step_number: int) -> str:
        """특정 스텝 번호의 파일 찾기"""
        files = self.list_experience_files()
        for file_info in files:
            if file_info["step_number"] == step_number:
                return file_info["filepath"]
        return ""
    
    def _cleanup_old_files(self):
        """오래된 파일 정리"""
        files = self.list_experience_files()
        if len(files) <= self.max_files:
            return
        
        # 생성 시간 기준으로 정렬
        files.sort(key=lambda x: x["created_time"])
        
        # 오래된 파일들 삭제
        files_to_delete = files[:-self.max_files]
        for file_info in files_to_delete:
            try:
                os.remove(file_info["filepath"])
                self.logger.info(f"🗑️ 오래된 경험 파일 삭제: {file_info['filename']}")
            except Exception as e:
                self.logger.warning(f"파일 삭제 실패 {file_info['filepath']}: {e}")
    
    def _get_file_size(self, filepath: str) -> str:
        """파일 크기를 읽기 좋은 형태로 반환"""
        try:
            size = os.path.getsize(filepath)
            if size < 1024:
                return f"{size} B"
            elif size < 1024 * 1024:
                return f"{size/1024:.1f} KB"
            elif size < 1024 * 1024 * 1024:
                return f"{size/(1024*1024):.1f} MB"
            else:
                return f"{size/(1024*1024*1024):.1f} GB"
        except:
            return "Unknown"
    
    def get_statistics(self) -> Dict[str, Any]:
        """경험 버퍼 파일들의 통계 반환"""
        files = self.list_experience_files()
        
        if not files:
            return {"total_files": 0, "total_size_mb": 0}
        
        total_size = sum(f["size"] for f in files)
        total_experiences = 0
        
        # 각 파일의 경험 개수 합계 (메타데이터에서 읽기)
        for file_info in files:
            try:
                _, metadata = self.load_experience_buffer(file_info["filepath"])
                total_experiences += metadata.get("num_experiences", 0)
            except:
                pass
        
        return {
            "total_files": len(files),
            "total_size_mb": total_size / (1024 * 1024),
            "total_experiences": total_experiences,
            "avg_experiences_per_file": total_experiences / len(files) if files else 0,
            "oldest_file": min(files, key=lambda x: x["created_time"])["filename"] if files else "",
            "newest_file": max(files, key=lambda x: x["created_time"])["filename"] if files else ""
        }
