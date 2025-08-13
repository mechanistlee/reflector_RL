"""
CAD Visualization Module
=======================

3D CAD 데이터 시각화를 위한 통합 모듈입니다.
STL 파일과 Point Cloud 데이터의 시각화를 지원합니다.

기존 visualize_pointcloud.py와 visualize_stl.py의 기능을 통합하여 제공합니다.
"""

import open3d as o3d
import numpy as np
from typing import Optional, Tuple, List


class CADVisualizer:
    """
    3D CAD 데이터 시각화를 위한 통합 클래스
    
    Features:
    - Point Cloud 시각화
    - STL 메시 시각화
    - 카메라 파라미터 저장/복원
    - 일관된 뷰 설정
    """
    
    def __init__(self):
        self._pcd_view_params = None
        self._stl_view_params = None
        
    def visualize_pointcloud(self, 
                           pointcloud: np.ndarray, 
                           vis: Optional[o3d.visualization.Visualizer] = None,
                           window_name: str = "PointCloud Viewer", 
                           point_size: int = 3) -> o3d.visualization.Visualizer:
        """
        Point Cloud 데이터를 시각화합니다.
        
        Args:
            pointcloud: N x 3 numpy array의 3D 점 데이터
            vis: 기존 visualizer 객체 (None이면 새로 생성)
            window_name: 윈도우 이름
            point_size: 점의 크기
            
        Returns:
            o3d.visualization.Visualizer 객체
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        
        if vis is None:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=window_name)
            vis.add_geometry(pcd)
            # 카메라 뷰 설정: (1,1,1) 방향에서 바라보도록 설정
            ctr = vis.get_view_control()
            ctr.set_front([1, 1, 1])
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, 0, 1])
            ctr.set_zoom(0.5)
            self._pcd_view_params = ctr.convert_to_pinhole_camera_parameters()
        else:
            vis.clear_geometries()
            vis.add_geometry(pcd)
            # 기존 카메라 파라미터를 복원
            if self._pcd_view_params is not None:
                ctr = vis.get_view_control()
                ctr.convert_from_pinhole_camera_parameters(self._pcd_view_params)
                
        opt = vis.get_render_option()
        opt.point_size = point_size
        vis.poll_events()
        vis.update_renderer()
        return vis
    
    def visualize_stl(self, 
                     stl_path: str, 
                     vis: Optional[o3d.visualization.Visualizer] = None,
                     window_name: str = "Reflector STL",
                     non_blocking: bool = True) -> o3d.visualization.Visualizer:
        """
        STL 파일을 시각화합니다.
        
        Args:
            stl_path: STL 파일 경로
            vis: 기존 visualizer 객체 (None이면 새로 생성)
            window_name: 윈도우 이름
            non_blocking: True면 비블로킹 모드로 실행
            
        Returns:
            o3d.visualization.Visualizer 객체
        """
        mesh = o3d.io.read_triangle_mesh(stl_path)
        mesh.compute_vertex_normals()
        
        if vis is None:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=window_name, visible=True)
            vis.add_geometry(mesh)
            # 카메라 뷰 설정
            ctr = vis.get_view_control()
            ctr.set_front([1, -1, 1])
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, 0, 1])
            ctr.set_zoom(0.6)
            self._stl_view_params = ctr.convert_to_pinhole_camera_parameters()
        else:
            vis.clear_geometries()
            vis.add_geometry(mesh)
            # 기존 카메라 파라미터를 복원
            if self._stl_view_params is not None:
                ctr = vis.get_view_control()
                ctr.convert_from_pinhole_camera_parameters(self._stl_view_params)
        
        if non_blocking:
            # 비블로킹 모드: 윈도우 업데이트만 하고 반환
            vis.poll_events()
            vis.update_renderer()
        else:
            # 기존 블로킹 모드
            vis.poll_events()
            vis.update_renderer()
            
        return vis
    
    def show_viewer(self, 
                   vis: o3d.visualization.Visualizer, 
                   viewer_type: str = "general") -> None:
        """
        시각화 뷰어를 실행합니다.
        
        Args:
            vis: visualizer 객체
            viewer_type: 뷰어 타입 ("pointcloud", "stl", "general")
        """
        if vis is not None:
            if viewer_type == "pointcloud":
                print("[PointCloud] Press Q or close window to exit viewer.")
            elif viewer_type == "stl":
                print("[INFO] 학습 종료 후 STL 뷰어를 닫으면 프로그램이 종료됩니다.")
            else:
                print("[INFO] Press Q or close window to exit viewer.")
            
            vis.run()
            vis.destroy_window()
    
    def show_pointcloud_viewer(self, vis: o3d.visualization.Visualizer) -> None:
        """Point Cloud 뷰어를 실행합니다 (하위 호환성)."""
        self.show_viewer(vis, "pointcloud")
    
    def show_stl_viewer(self, vis: o3d.visualization.Visualizer) -> None:
        """STL 뷰어를 실행합니다 (하위 호환성)."""
        self.show_viewer(vis, "stl")


# 전역 인스턴스
_cad_visualizer = CADVisualizer()

# 하위 호환성을 위한 함수들 (기존 코드와의 호환성 유지)
def visualize_pointcloud(pointcloud: np.ndarray, 
                        vis: Optional[o3d.visualization.Visualizer] = None,
                        window_name: str = "PointCloud Viewer", 
                        point_size: int = 3) -> o3d.visualization.Visualizer:
    """
    Point Cloud 시각화 함수 (하위 호환성)
    
    기존 visualize_pointcloud.py의 함수와 동일한 인터페이스를 제공합니다.
    """
    return _cad_visualizer.visualize_pointcloud(pointcloud, vis, window_name, point_size)


def show_pointcloud_viewer(vis: o3d.visualization.Visualizer) -> None:
    """Point Cloud 뷰어 실행 함수 (하위 호환성)"""
    _cad_visualizer.show_pointcloud_viewer(vis)


def visualize_stl(stl_path: str, 
                 vis: Optional[o3d.visualization.Visualizer] = None,
                 non_blocking: bool = True) -> o3d.visualization.Visualizer:
    """
    STL 시각화 함수 (하위 호환성)
    
    기존 visualize_stl.py의 함수와 동일한 인터페이스를 제공합니다.
    """
    return _cad_visualizer.visualize_stl(stl_path, vis, non_blocking=non_blocking)


def show_viewer(vis: o3d.visualization.Visualizer) -> None:
    """STL 뷰어 실행 함수 (하위 호환성)"""
    _cad_visualizer.show_stl_viewer(vis)


# 모듈 전역 변수 (기존 코드와의 호환성)
_pcd_view_params = None


__all__ = [
    'CADVisualizer',
    'visualize_pointcloud',
    'show_pointcloud_viewer', 
    'visualize_stl',
    'show_viewer'
]
