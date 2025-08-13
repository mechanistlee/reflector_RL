"""
SPEOS Integration Module (Unified Class-Based Structure)
=======================================================

스페오스와의 임시 연동을 위한 통합 클래스 기반 모듈입니다.
완전한 클래스 중심 구조로 다음 기능들을 제공합니다:
- SPEOS 시뮬레이션 실행 및 제어
- XMP 파일 변환 및 검증
- 포인트클라우드 생성 및 처리
- 메쉬 생성 및 STL 변환
- 광학 효율 계산
"""

import os
import time
import struct
import subprocess
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple
import open3d as o3d
import win32com.client


class SpeosUtility:
    """
    통합 SPEOS 유틸리티 클래스
    
    SPEOS 광학 시뮬레이션과의 연동을 위한 모든 기능을 제공하는 통합 클래스입니다.
    파일 모니터링, XMP 변환, 포인트클라우드 처리, 메쉬 생성, 효율 계산 등을 포함합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        SPEOS 유틸리티 초기화
        
        Args:
            config: SPEOS 연동 설정 딕셔너리
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 필수 경로들 검증
        self._validate_config()
    
    def _validate_config(self):
        """설정 검증"""
        required_keys = [
            'control_file_path', 'xmp_file_path', 'txt_output_path',
            'grid_rows', 'grid_cols'
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"필수 설정 키가 없습니다: {key}")
        
        # 기본값 설정
        defaults = {
            'led_output': 100.0,
            'flip_updown': False,
            'flip_leftright': False,
            'spacing_x': 1.0,
            'spacing_y': 1.0,
            'x_min': 0.0,
            'y_min': 0.0,
            'z_init': 0.0
        }
        
        for key, default_value in defaults.items():
            if key not in self.config:
                self.config[key] = default_value
    
    def wait_for_file_update(self, file_path: Optional[str] = None, 
                           timeout: int = 120, check_interval: float = 1.0) -> bool:
        """파일이 업데이트될 때까지 대기"""
        if file_path is None:
            file_path = self.config['xmp_file_path']
        
        # 이미 절대 경로인 경우 중복 변환 방지
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        
        self.logger.info(f"파일 존재 확인: {file_path}")
        
        # 파일 존재 여부를 정확히 확인
        file_exists = os.path.exists(file_path)
        self.logger.info(f"파일 존재 상태: {file_exists}")
        
        if not file_exists:
            self.logger.warning(f"파일이 존재하지 않음: {file_path}")
            # 파일이 생성될 때까지 대기 (최대 30초)
            self.logger.info("파일 생성 대기 중...")
            creation_start = time.time()
            while time.time() - creation_start < 30:
                if os.path.exists(file_path):
                    self.logger.info(f"파일 생성 감지됨: {file_path}")
                    file_exists = True
                    break
                time.sleep(1.0)
            
            if not file_exists:
                self.logger.error(f"파일 생성 타임아웃: {file_path}")
                return False
        
        # 초기 파일 수정 시간 기록
        try:
            initial_mtime = os.path.getmtime(file_path)
            initial_time_str = time.ctime(initial_mtime)
            self.logger.info(f"초기 파일 수정 시간: {initial_time_str}")
        except OSError as e:
            self.logger.error(f"파일 접근 오류: {e}")
            return False
        
        self.logger.info(f"파일 업데이트 대기 시작 (타임아웃: {timeout}초)")
        
        start_time = time.time()
        check_count = 0
        
        while time.time() - start_time < timeout:
            try:
                current_mtime = os.path.getmtime(file_path)
                check_count += 1
                
                # 파일이 업데이트되었는지 확인
                if current_mtime > initial_mtime:
                    current_time_str = time.ctime(current_mtime)
                    elapsed = time.time() - start_time
                    self.logger.info(f"파일 업데이트 감지됨! (경과 시간: {elapsed:.1f}초)")
                    self.logger.info(f"업데이트된 수정 시간: {current_time_str}")
                    return True
                
                # 주기적으로 상태 로그 출력
                if check_count % 10 == 0:
                    elapsed = time.time() - start_time
                    self.logger.info(f"대기 중... ({elapsed:.1f}초 경과, {check_count}회 확인)")
                    
                time.sleep(check_interval)
                
            except OSError as e:
                self.logger.error(f"파일 모니터링 오류: {e}")
                return False
        
        # 타임아웃 발생
        elapsed = time.time() - start_time
        self.logger.warning(f"파일 업데이트 타임아웃: {file_path} ({elapsed:.1f}초 경과, {check_count}회 확인)")
        return False
    
    def generate_origin_pointcloud(self, grid_rows: Optional[int] = None, 
                                 grid_cols: Optional[int] = None,
                                 spacing_x: Optional[float] = None,
                                 spacing_y: Optional[float] = None,
                                 x_min: Optional[float] = None,
                                 y_min: Optional[float] = None,
                                 z_init: Optional[float] = None) -> np.ndarray:
        """기본 포인트클라우드 생성"""
        # 기본값 설정 - 우선순위: 매개변수 > SpeosConfig 스타일 > 기존 스타일 > 기본값
        if grid_rows is None:
            grid_rows = self.config['grid_rows']
        if grid_cols is None:
            grid_cols = self.config['grid_cols']
        if spacing_x is None:
            # 🎯 SpeosConfig의 grid_cell_size_x 우선 사용
            spacing_x = self.config.get('grid_cell_size_x', self.config.get('spacing_x', 1.0))
        if spacing_y is None:
            # 🎯 SpeosConfig의 grid_cell_size_y 우선 사용
            spacing_y = self.config.get('grid_cell_size_y', self.config.get('spacing_y', 1.0))
        if x_min is None:
            x_min = self.config.get('grid_origin_x', self.config.get('x_min', 0.0))
        if y_min is None:
            y_min = self.config.get('grid_origin_y', self.config.get('y_min', 0.0))
        if z_init is None:
            z_init = self.config.get('grid_origin_z', self.config.get('z_init', 0.0))
        
        # 그리드 생성
        x_coords = np.linspace(x_min, x_min + (grid_cols - 1) * spacing_x, grid_cols)
        y_coords = np.linspace(y_min, y_min + (grid_rows - 1) * spacing_y, grid_rows)
        
        # 메쉬그리드 생성
        X, Y = np.meshgrid(x_coords, y_coords)
        Z = np.full_like(X, z_init)
        
        # 포인트클라우드 배열 생성
        pointcloud = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        
        self.logger.info(f"포인트클라우드 생성 완료: {pointcloud.shape}, 그리드 1칸 크기: {spacing_x:.3f}x{spacing_y:.3f}mm")
        return pointcloud


def create_speos_config(control_file_path: str, xmp_file_path: str, txt_output_path: str,
                    ply_output_path: str = None, stl_output_path: str = None, 
                    freecad_cmd_path: str = None,
                    grid_rows: int = 10, grid_cols: int = 10, led_output: float = 100.0,
                    **kwargs) -> Dict[str, Any]:
    """SPEOS 연동 설정 딕셔너리 생성 편의 함수"""
    config = {
        'control_file_path': control_file_path,
        'xmp_file_path': xmp_file_path,
        'txt_output_path': txt_output_path,
        'ply_output_path': ply_output_path,
        'stl_output_path': stl_output_path,
        'freecad_cmd_path': freecad_cmd_path,
        'grid_rows': grid_rows,
        'grid_cols': grid_cols,
        'led_output': led_output,
        'flip_updown': False,
        'flip_leftright': False,
        'spacing_x': 1.0,
        'spacing_y': 1.0,
        'x_min': 0.0,
        'y_min': 0.0,
        'z_init': 0.0
    }
    config.update(kwargs)
    return config


def xmp_to_txt(xmp_path, txt_path, flip_updown=True, flip_leftright=False):
    # 절대 경로로 변환
    xmp_path = os.path.abspath(xmp_path)
    txt_path = os.path.abspath(txt_path)
    tmp_raw_path = txt_path + ".raw"

    # XMP Viewer COM 객체 생성
    VPL = win32com.client.Dispatch("XmpViewer.Application")

    # XMP 파일 열기
    result = VPL.OpenFile(xmp_path)
    if result != 1:
        print("X File open failed")
        raise RuntimeError("X File open failed")

    export_result = VPL.ExportTXT(tmp_raw_path)
    if export_result == 0:
        print("X TXT export failed")
        raise RuntimeError("X TXT export failed")

    # ▼ "x y value" 이후 데이터만 읽기
    with open(tmp_raw_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if "x" in line.lower() and "y" in line.lower() and "value" in line.lower():
            start_idx = i + 1
            break
    if start_idx is None:
        print("X 'x y value' 구간을 찾을 수 없습니다.")
        raise RuntimeError("X 'x y value' 구간을 찾을 수 없습니다.")

    # ▼ 값 파싱
    data = []
    x_set = set()
    y_set = set()
    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) == 3:
            x, y, v = map(float, parts)
            data.append((x, y, v))
            x_set.add(x)
            y_set.add(y)
    x_list = sorted(list(x_set))
    y_list = sorted(list(y_set))
    x_index = {x: i for i, x in enumerate(x_list)}
    y_index = {y: i for i, y in enumerate(y_list)}
    value_map = np.zeros((len(y_list), len(x_list)))
    for x, y, v in data:
        i = y_index[y]  # 행
        j = x_index[x]  # 열
        value_map[i, j] = v
    # flip 적용
    if flip_updown:
        value_map = np.flipud(value_map)
    if flip_leftright:
        value_map = np.fliplr(value_map)
    # 결과 저장
    np.savetxt(txt_path, value_map, fmt="%.6f", delimiter="\t")
    # 임시 파일 삭제
    os.remove(tmp_raw_path)
    
    # ✅ 중요: value_map 반환 추가!
    return value_map
    
def pointcloud_to_stl(pointcloud, stl_output_path: str, 
                      ply_output_path: Optional[str] = None,
                      freecad_cmd_path: Optional[str] = None,
                      poisson_depth: int = 9) -> bool:
    """
    포인트클라우드를 STL 메쉬 파일로 변환하는 통합 함수
    
    Args:
        pointcloud: 입력 포인트클라우드 (numpy array 또는 file path 또는 open3d PointCloud)
        stl_output_path: 출력할 STL 파일 경로
        ply_output_path: 중간 PLY 파일 경로 (기본값: STL 경로에서 확장자만 변경)
        freecad_cmd_path: FreeCAD 실행 파일 경로 (기본값: 시스템 PATH에서 찾기)
        poisson_depth: Poisson 메쉬 생성 깊이 (기본값: 9)
    
    Returns:
        bool: 변환 성공 여부
    """
    try:
        # 1. 포인트클라우드 로드 및 변환
        if isinstance(pointcloud, str):
            # 파일 경로인 경우 파일에서 로드
            ext = os.path.splitext(pointcloud)[1].lower()
            if ext in ['.ply', '.pcd', '.xyz', '.xyzn', '.xyzrgb', '.pts']:
                pcd = o3d.io.read_point_cloud(pointcloud)
            elif ext in ['.txt', '.csv']:
                data = np.loadtxt(pointcloud, delimiter=None)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(data[:, :3])
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {ext}")
        elif isinstance(pointcloud, np.ndarray):
            # numpy 배열인 경우 open3d PointCloud로 변환
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
        elif hasattr(pointcloud, 'points'):
            # 이미 open3d PointCloud 객체인 경우
            pcd = pointcloud
        else:
            raise ValueError("포인트클라우드는 파일 경로, numpy 배열, 또는 open3d PointCloud 객체여야 합니다")
        
        # 2. 포인트 클라우드 검증
        if len(pcd.points) == 0:
            raise ValueError("포인트클라우드가 비어있습니다")
        
        # 3. 법선 벡터 추정 및 Poisson 메쉬 생성
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)
        
        # 4. 메쉬 정리 (중복 제거, 퇴화 삼각형 제거 등)
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        mesh.remove_non_manifold_edges()
        
        # 메쉬 법선 벡터 계산 (STL 저장을 위해 필수)
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        
        # 5. PLY 파일 경로 설정
        if ply_output_path is None:
            ply_output_path = os.path.splitext(stl_output_path)[0] + ".ply"
        
        # 6. 출력 디렉토리 생성
        os.makedirs(os.path.dirname(stl_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(ply_output_path), exist_ok=True)
        
        # 7. PLY 파일로 메쉬 저장
        success = o3d.io.write_triangle_mesh(ply_output_path, mesh)
        if not success:
            raise RuntimeError(f"PLY 파일 저장 실패: {ply_output_path}")
        
        # 8. FreeCAD를 사용하여 STL로 변환
        convert_script_path = os.path.join(os.path.dirname(ply_output_path), "convert_ply_to_stl.py")
        converter_script = f"""# -*- coding: utf-8 -*-
import Mesh
import FreeCAD

try:
    FreeCAD.newDocument()
    mesh_obj = Mesh.Mesh(r\"{ply_output_path}\")
    doc = FreeCAD.ActiveDocument
    mesh_feature = doc.addObject(\"Mesh::Feature\", \"MeshObj\")
    mesh_feature.Mesh = mesh_obj
    mesh_obj.write(r\"{stl_output_path}\")
    print(\"STL export completed successfully.\")
except Exception as e:
    print(f\"Error during STL conversion: {{e}}\")
    exit(1)
"""
        
        with open(convert_script_path, "w", encoding='utf-8') as f:
            f.write(converter_script)
        
        # 9. FreeCAD 실행
        freecad_commands = [
            freecad_cmd_path if freecad_cmd_path else "freecad",
            "FreeCAD",
            "freecad-daily"
        ]
        
        conversion_success = False
        for cmd in freecad_commands:
            if cmd is None:
                continue
            try:
                result = subprocess.run(
                    [cmd, convert_script_path], 
                    check=True, 
                    capture_output=True, 
                    text=True,
                    timeout=60
                )
                if "STL export completed successfully" in result.stdout:
                    conversion_success = True
                    break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        # 10. 임시 파일 정리
        try:
            os.remove(convert_script_path)
        except:
            pass
        
        if not conversion_success:
            # FreeCAD를 사용할 수 없는 경우 Open3D로 직접 STL 저장 시도
            try:
                # STL 저장 전에 메쉬 법선 계산 (필수)
                mesh.compute_vertex_normals()
                mesh.compute_triangle_normals()
                
                success = o3d.io.write_triangle_mesh(stl_output_path, mesh)
                if success:
                    print(f"[INFO] Open3D를 사용하여 STL 파일 생성: {stl_output_path}")
                    return True
                else:
                    raise RuntimeError("Open3D STL 저장 실패")
            except Exception as e:
                print(f"[ERROR] STL 변환 실패 (FreeCAD 및 Open3D): {e}")
                return False
        else:
            print(f"[INFO] FreeCAD를 사용하여 STL 파일 생성: {stl_output_path}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] pointcloud_to_stl 변환 중 오류 발생: {e}")
        return False




if __name__ == "__main__":
    print("SPEOS Utility Module - 테스트 모드")
    
    # 기본 설정 생성
    config = {
        'control_file_path': "test_control.txt",
        'xmp_file_path': "test.xmp",
        'txt_output_path': "test_output.txt",
        'grid_rows': 5,
        'grid_cols': 5
    }
    
    # SPEOS 유틸리티 객체 생성
    speos = SpeosUtility(config)
    
    # 기본 포인트클라우드 생성 테스트
    pointcloud = speos.generate_origin_pointcloud()
    print(f"포인트클라우드 생성 테스트: {pointcloud.shape}")
    
    print("테스트 완료!")

def wait_for_file_update(file_path: str, timeout: int = 120, check_interval: float = 1.0) -> bool:
    """
    파일이 업데이트될 때까지 대기 (하위 호환성을 위한 전역 함수)
    """
    # 임시 config로 SpeosUtility 객체 생성
    temp_config = {
        'control_file_path': '',
        'xmp_file_path': file_path,
        'txt_output_path': '',
        'grid_rows': 10,
        'grid_cols': 10
    }
    utility = SpeosUtility(temp_config)
    return utility.wait_for_file_update(file_path, timeout, check_interval)


def generate_origin_pointcloud(grid_rows: int = 10, grid_cols: int = 10,
                              spacing_x: float = 1.0, spacing_y: float = 1.0,
                              x_min: float = 0.0, y_min: float = 0.0, z_init: float = 0.0) -> np.ndarray:
    """
    기본 포인트클라우드 생성 (하위 호환성을 위한 전역 함수)
    """
    temp_config = {
        'control_file_path': '',
        'xmp_file_path': '',
        'txt_output_path': '',
        'grid_rows': grid_rows,
        'grid_cols': grid_cols,
        'spacing_x': spacing_x,
        'spacing_y': spacing_y,
        'x_min': x_min,
        'y_min': y_min,
        'z_init': z_init
    }
    utility = SpeosUtility(temp_config)
    return utility.generate_origin_pointcloud()
