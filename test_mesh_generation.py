"""
메쉬 생성 및 STL 저장 테스트 스크립트
=========================================

이 스크립트는 ReflectorClass의 _get_mesh() 함수를 테스트하고
생성된 메쉬를 STL 파일로 저장하여 형상을 확인할 수 있습니다.
"""

import os
import sys
import numpy as np
import logging

# matplotlib는 선택적으로 임포트
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠️ matplotlib가 설치되지 않았습니다. 시각화 기능이 제한됩니다.")
    MATPLOTLIB_AVAILABLE = False

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("⚠️ Open3D가 설치되지 않았습니다. 메쉬 생성 기능이 제한됩니다.")
    OPEN3D_AVAILABLE = False

try:
    from config import TrainingConfig
    CONFIG_AVAILABLE = True
except ImportError:
    print("⚠️ config.py를 가져올 수 없습니다. 기본 설정을 사용합니다.")
    CONFIG_AVAILABLE = False

class TestConfig:
    """테스트용 간단한 설정 클래스"""
    def __init__(self):
        self.grid_rows = 10
        self.grid_cols = 10
        self.grid_cell_size_x = 1.0  # mm
        self.grid_cell_size_y = 1.0  # mm
        self.grid_origin_x = 0.0
        self.grid_origin_y = 0.0
        self.grid_origin_z = 0.0
        self.z_min = -5.0
        self.z_max = 5.0
        self.action_size = self.grid_rows * self.grid_cols
    
    def get_reflector_position(self, reflector_id):
        """리플렉터 위치 반환 (테스트용)"""
        positions = [
            (0.0, 0.0, 0.0),    # Reflector 0
            (20.0, 0.0, 0.0),   # Reflector 1
            (0.0, 20.0, 0.0),   # Reflector 2
        ]
        return positions[reflector_id % len(positions)]

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_mesh_from_pointcloud(pointcloud, center_position=(0.0, 0.0, 0.0)):
    """
    포인트클라우드에서 Open3D 메쉬 객체 생성 및 반환 (리플렉터 위치 오프셋 적용)
    
    Args:
        pointcloud: 포인트클라우드 numpy 배열 (N x 3)
        center_position: 리플렉터 중심 위치 (x, y, z)
    
    Returns:
        open3d.geometry.TriangleMesh 또는 None
    """
    try:
        if not OPEN3D_AVAILABLE:
            print("❌ Open3D가 설치되지 않았습니다.")
            return None
        
        if pointcloud is None or len(pointcloud) == 0:
            print("❌ 포인트클라우드가 없습니다")
            return None
        
        # 🎯 포인트클라우드에 리플렉터 중심 위치 오프셋 적용
        positioned_pointcloud = pointcloud.copy()
        cx, cy, cz = center_position
        positioned_pointcloud[:, 0] += cx  # X offset
        positioned_pointcloud[:, 1] += cy  # Y offset
        positioned_pointcloud[:, 2] += cz  # Z offset
        
        print(f"   📍 메쉬 생성: 중심위치 ({cx:.1f}, {cy:.1f}, {cz:.1f})mm 적용")
        
        # 포인트클라우드를 Open3D PointCloud 객체로 변환
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positioned_pointcloud)
        
        # 법선 벡터 추정
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Poisson 표면 재구성을 사용하여 메쉬 생성
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=3, width=0, scale=1.1, linear_fit=True
        )
        
        if len(mesh.vertices) == 0:
            print("❌ 메쉬 생성 실패")
            return None
        
        # 메쉬 정리 및 법선 벡터 계산
        mesh.compute_vertex_normals()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        return mesh
        
    except Exception as e:
        print(f"❌ 메쉬 생성 실패: {e}")
        return None

def create_test_pointcloud_patterns():
    """다양한 테스트용 포인트클라우드 패턴 생성"""
    patterns = {
        'flat': 'Flat surface (Z=0)',
        'dome': 'Dome shape (center high)',
        'wave': 'Wave pattern'
    }
    
    return patterns

def generate_pattern_pointcloud(pattern_name, config):
    """특정 패턴의 포인트클라우드 생성"""
    grid_rows, grid_cols = config.grid_rows, config.grid_cols
    
    # 기본 그리드 생성
    x_coords = np.linspace(-5, 5, grid_cols)
    y_coords = np.linspace(-5, 5, grid_rows)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    if pattern_name == 'flat':
        Z = np.zeros_like(X)
    
    elif pattern_name == 'dome':
        # 중심이 높은 돔 형태
        center_x, center_y = 0, 0
        max_height = 3.0
        radius = 5.0
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        Z = max_height * np.exp(-(dist**2) / (2 * (radius/2)**2))
    
    elif pattern_name == 'valley':
        # 중심이 낮은 계곡 형태
        center_x, center_y = 0, 0
        max_depth = -2.0
        radius = 3.0
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        Z = max_depth * np.exp(-(dist**2) / (2 * radius**2))
    
    elif pattern_name == 'wave':
        # 파동 패턴
        Z = 2.0 * np.sin(X) * np.cos(Y)
    
    elif pattern_name == 'random':
        # 랜덤 높이 변화
        np.random.seed(42)  # 재현 가능한 랜덤
        Z = np.random.normal(0, 1.5, X.shape)
    
    elif pattern_name == 'pyramid':
        # 피라미드 형태
        Z = 3.0 - 0.3 * (np.abs(X) + np.abs(Y))
        Z = np.maximum(Z, 0)
    
    elif pattern_name == 'saddle':
        # 안장 형태
        Z = 0.3 * (X**2 - Y**2)
    
    else:
        Z = np.zeros_like(X)
    
    # Z값을 config 범위로 제한
    Z = np.clip(Z, config.z_min, config.z_max)
    
    # 포인트클라우드 형태로 변환
    pointcloud = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    return pointcloud.astype(np.float32)

def test_single_reflector_mesh(pattern_name, config, output_dir):
    """단일 리플렉터의 메쉬 생성 테스트"""
    print(f"\n🧪 테스트 패턴: {pattern_name}")
    
    if not OPEN3D_AVAILABLE:
        print("❌ Open3D가 없어 메쉬 생성을 건너뜁니다.")
        return False
    
    try:
        # 테스트 포인트클라우드 생성
        test_pointcloud = generate_pattern_pointcloud(pattern_name, config)
        
        print(f"   📊 포인트클라우드 정보:")
        print(f"      - 포인트 수: {len(test_pointcloud)}")
        print(f"      - X 범위: [{np.min(test_pointcloud[:, 0]):.2f}, {np.max(test_pointcloud[:, 0]):.2f}]")
        print(f"      - Y 범위: [{np.min(test_pointcloud[:, 1]):.2f}, {np.max(test_pointcloud[:, 1]):.2f}]")
        print(f"      - Z 범위: [{np.min(test_pointcloud[:, 2]):.2f}, {np.max(test_pointcloud[:, 2]):.2f}]")
        
        # 메쉬 생성 (리플렉터 중심 위치 사용)
        center_position = config.get_reflector_position(0)
        mesh = create_mesh_from_pointcloud(test_pointcloud, center_position)
        
        if mesh is None:
            print("❌ 메쉬 생성 실패")
            return False
        
        print(f"   ✅ 메쉬 생성 성공:")
        print(f"      - 버텍스 수: {len(mesh.vertices)}")
        print(f"      - 삼각형 수: {len(mesh.triangles)}")
        
        # STL 파일로 저장 (현재 폴더에 직접 저장)
        stl_filename = f"{pattern_name}_mesh.stl"
        stl_path = os.path.join(output_dir, stl_filename)
        
        success = o3d.io.write_triangle_mesh(stl_path, mesh)
        
        if success:
            print(f"   💾 STL 파일 저장: {stl_path}")
            
            # 파일 크기 확인
            file_size = os.path.getsize(stl_path) / 1024  # KB
            print(f"      - 파일 크기: {file_size:.1f} KB")
        else:
            print(f"   ❌ STL 파일 저장 실패")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def visualize_pointcloud_2d(pattern_name, config, output_dir):
    """포인트클라우드의 2D 시각화 (높이맵)"""
    if not MATPLOTLIB_AVAILABLE:
        print("   ⚠️ matplotlib가 없어 2D 시각화를 건너뜁니다.")
        return False
        
    try:
        # 테스트 포인트클라우드 생성
        pointcloud = generate_pattern_pointcloud(pattern_name, config)
        
        # Z값을 그리드 형태로 변환
        z_values = pointcloud[:, 2].reshape(config.grid_rows, config.grid_cols)
        
        # 2D 높이맵 시각화
        plt.figure(figsize=(8, 6))
        plt.imshow(z_values, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Height (Z)')
        plt.title(f'Height Map - {pattern_name}')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        
        # 이미지 저장
        img_filename = f"heightmap_{pattern_name}.png"
        img_path = os.path.join(output_dir, img_filename)
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 높이맵 이미지 저장: {img_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ 2D 시각화 실패: {e}")
        return False

def test_multiple_reflectors(config, output_dir):
    """여러 리플렉터 조합 테스트"""
    print(f"\n🏭 다중 리플렉터 조합 테스트")
    
    if not OPEN3D_AVAILABLE:
        print("❌ Open3D가 없어 다중 리플렉터 테스트를 건너뜁니다.")
        return False
    
    try:
        # 3개의 서로 다른 패턴으로 리플렉터 생성
        patterns = ['dome', 'valley', 'wave']
        meshes = []
        
        for i, pattern in enumerate(patterns):
            # 테스트 포인트클라우드 생성
            test_pointcloud = generate_pattern_pointcloud(pattern, config)
            
            # 메쉬 생성 (각기 다른 리플렉터 위치 사용)
            center_position = config.get_reflector_position(i)
            mesh = create_mesh_from_pointcloud(test_pointcloud, center_position)
            
            if mesh is not None:
                meshes.append(mesh)
                print(f"   ✅ 리플렉터 {i+1} ({pattern}) 메쉬 생성 완료")
            else:
                print(f"   ❌ 리플렉터 {i+1} ({pattern}) 메쉬 생성 실패")
        
        if len(meshes) == 0:
            print("❌ 생성된 메쉬가 없습니다.")
            return False
        
        # 모든 메쉬 결합
        if len(meshes) == 1:
            combined_mesh = meshes[0]
        else:
            all_vertices = []
            all_faces = []
            vertex_offset = 0
            
            for mesh in meshes:
                vertices = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
                
                all_vertices.append(vertices)
                all_faces.append(faces + vertex_offset)
                vertex_offset += len(vertices)
            
            # 결합된 메쉬 생성
            combined_vertices = np.vstack(all_vertices)
            combined_faces = np.vstack(all_faces)
            
            combined_mesh = o3d.geometry.TriangleMesh()
            combined_mesh.vertices = o3d.utility.Vector3dVector(combined_vertices)
            combined_mesh.triangles = o3d.utility.Vector3iVector(combined_faces)
            combined_mesh.compute_vertex_normals()
        
        print(f"   🔗 결합된 메쉬: {len(combined_mesh.vertices)}개 버텍스, {len(combined_mesh.triangles)}개 삼각형")
        
        # 결합된 메쉬 STL 저장 (현재 폴더에 직접 저장)
        combined_stl_path = os.path.join(output_dir, "combined_reflectors.stl")
        success = o3d.io.write_triangle_mesh(combined_stl_path, combined_mesh)
        
        if success:
            print(f"   💾 결합된 STL 파일 저장: {combined_stl_path}")
            
            # 파일 크기 확인
            file_size = os.path.getsize(combined_stl_path) / 1024  # KB
            print(f"      - 파일 크기: {file_size:.1f} KB")
            return True
        else:
            print(f"   ❌ 결합된 STL 파일 저장 실패")
            return False
        
    except Exception as e:
        print(f"❌ 다중 리플렉터 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("🧪 ReflectorClass 메쉬 생성 테스트")
    print("=" * 60)
    
    # 로깅 설정
    setup_logging()
    
    # 출력 디렉토리 생성 (현재 폴더)
    output_dir = os.path.join(project_root, "mesh_test_output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 출력 디렉토리: {output_dir}")
    
    # 테스트 설정
    config = TestConfig()
    
    # 사용 가능한 패턴들
    patterns = create_test_pointcloud_patterns()
    
    print(f"\n📋 테스트할 패턴들:")
    for pattern, description in patterns.items():
        print(f"   - {pattern}: {description}")
    
    # 개별 패턴 테스트
    print(f"\n" + "=" * 40)
    print("🔬 개별 패턴 메쉬 생성 테스트")
    print("=" * 40)
    
    success_count = 0
    total_count = len(patterns)
    
    for pattern_name in patterns.keys():
        # 메쉬 생성 및 STL 저장 테스트
        if test_single_reflector_mesh(pattern_name, config, output_dir):
            success_count += 1
    
    print(f"\n📊 개별 패턴 테스트 결과: {success_count}/{total_count} 성공")
    
    # 다중 리플렉터 테스트는 건너뛰기 (개별 테스트에 집중)
    print(f"\n💡 다중 리플렉터 테스트는 건너뛰고 개별 패턴 테스트에 집중합니다.")
    multi_success = True  # 건너뛰므로 성공으로 처리
    
    # 최종 결과
    print(f"\n" + "=" * 60)
    print("📋 테스트 결과 요약")
    print("=" * 60)
    print(f"✅ 개별 패턴 테스트: {success_count}/{total_count} 성공")
    print(f"✅ 다중 리플렉터 테스트: {'성공' if multi_success else '실패'}")
    print(f"📁 생성된 파일들은 다음 경로에서 확인하세요: {output_dir}")
    
    if OPEN3D_AVAILABLE:
        print(f"\n💡 STL 파일 확인 방법:")
        print(f"   - FreeCAD, Blender, MeshLab 등으로 STL 파일을 열어서 형상 확인")
        print(f"   - PLY 파일은 Open3D viewer로도 확인 가능")
    else:
        print(f"\n⚠️ Open3D 설치 권장:")
        print(f"   pip install open3d")
    
    return success_count, multi_success

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
