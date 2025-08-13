





from System.IO import Path
import time
import os

StructureData = "../CAD/Proto3D.CATPart"
ReflectorData = "../CAD/mesh/Reflector001.stl"
SpeosControl = "../env/SpeosControl.txt"

MaterialReflector = None
directSimulation = None

# 바디 이름으로 찾기
def GetBodyName(target_name):
    for comp in GetRootPart().Components:
        for body in comp.GetBodies():
            if body.Name == target_name:
                return body
    return None

# 컴포넌트 이름으로 찾기
def GetCompoenetName(target_name):
    for comp in GetRootPart().Components:
        if comp.GetName() == target_name:
            return comp
    return None

def AssignMaterial():
    found_names = []

    for i in range(1, 6):
        name = "Reflector{:03d}".format(i)
        target_obj = GetCompoenetName(name)

        if target_obj and M_Reflector:
            M_Reflector.VolumeGeometries.Set(target_obj)
            found_names.append(name)


def run_simulation_step(reflector_file_base_path):
    base_dir = os.path.dirname(reflector_file_base_path)

    # ✅ 1. Reflector001 ~ Reflector100 삭제
    for i in range(1, 6):
        name = "Reflector{:03d}".format(i)
        comp = GetCompoenetName(name)
        if comp:
            comp.Delete()
            print("🗑️ Deleted {}".format(name))

    # ✅ 2. Reflector001.stl ~ Reflector100.stl 불러오기
    for i in range(1, 6):
        name = "Reflector{:03d}".format(i)
        filepath = os.path.join(base_dir, "{}.stl".format(name))

        wait_count = 0
        while not os.path.exists(filepath):
            if wait_count % 10 == 0:
                print("⏳ Waiting for {}.stl...".format(name))
            time.sleep(0.25)
            wait_count += 1

        SpeosAsm.CADUpdate.Import(filepath)

    AssignMaterial()
    directSimulation.Geometries.SelectAll()
    directSimulation.Compute()



M_Reflector = SpeosSim.Material.Find("M_Reflector") 

target_body_material = {
    "Reflector": ("component", M_Reflector)
}

directSimulation = SpeosSim.SimulationDirect.Create()
directSimulation.Geometries.SelectAll()
directSimulation.Sources.SelectAll()
directSimulation.Sensors.SelectAll()
directSimulation.NbRays = 1000000


# ✅ 반복 루프 조건 확인 및 실행 (2번째 줄이 1일 때 반복 실행)
while True:
    with open(SpeosControl, "r") as f:
        flag = f.read().strip()

    if flag == "1":
        run_simulation_step(ReflectorData)
        print("✅ Simulation executed from control flag.")
    elif flag == "0":
        print("❌ Control flag is 0. Waiting...")
    else:
        print("🛑 Invalid control flag. Stopping loop.")
        break

    time.sleep(0.5)
