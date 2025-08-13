from System.IO import Path
import time



SpeosAsm.CADUpdate.Import(StructureData) # CAD 데이터 가져오기




Bodies = GetRootPart().GetAllBodies() # 모든 바디 객체 가져오기
Body = GetRootPart().Components[0].GetBodies()[0] # 0번째 컴포넌트의 0번째 바디 지오메트리 객체 생성
Body = GetRootPart().Components[0].GetBodies("HSG") # 0번째 컴포넌트의 HSG 바디 지오메트리 객체 생성
Body = GetRootPart().GetBodies("HSG") # 전체 컴포넌트에서 HSG 바디 지오메트리 객체 생성


MaterialHSG = SpeosSim.Material.Create() # MaterialHSG 물성 객체 생성
MaterialHSG.Name = "HSG" # MaterialHSG 이름 설정
MaterialHSG = SpeosSim.Material.Find("HSG") # HSG 물성 객체 가져오기
MaterialHSG.VolumeGeometries.Set(GeoHSG) # MaterialHSG 물성 객체에 HSG 지오메트리 할당


directSimulation = SpeosSim.SimulationDirect.Create() # 직선 시뮬레이션 객체 생성
directSimulation.Geometries.SelectAll() # 모든 지오메트리 선택
directSimulation.Sources.SelectAll() # 모든 광원 선택
directSimulation.Sensors.SelectAll() # 모든 센서 선택
directSimulation.NbRays = 10000000 # 광선 수 설정


지오메트리 리스트 {HSG, LENS, LED, Reflector} 
물성 리스트 {M_HSG, M_LENS, M_ED, M_Reflector}


물성 리스트 출력 매서드 없음.



