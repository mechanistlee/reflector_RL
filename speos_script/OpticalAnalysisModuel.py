from System.IO import Path
import time
import os

StructureData = "../CAD/Proto3D.CATPart"
ReflectorData = "../CAD/Reflector.stl"
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

    for name, (target_type, mat) in target_body_material.items():
        target_obj = None

        if target_type == "body":
            target_obj = GetBodyName(name)
        elif target_type == "component":
            target_obj = GetCompoenetName(name)

        if target_obj:
            if mat:
                mat.VolumeGeometries.Set(target_obj)
    print("✅ 최종 할당된 대상들:", found_names)

def OpticalAnalysisSetup(StructureData, ReflectorData):
    global MaterialReflector, directSimulation, M_HSG, M_LENS, M_LED, M_Reflector

    SpeosAsm.CADUpdate.Import(StructureData)
    SpeosAsm.CADUpdate.Import(ReflectorData)

    M_HSG = SpeosSim.Material.Create()
    M_HSG.Name = "M_HSG"
    M_HSG.OpticalPropertiesType = SpeosSim.Material.EnumOpticalPropertiesType.Volumic
    M_HSG.SOPType = SpeosSim.Material.EnumSOPType.Mirror
    M_HSG.SOPReflectance = 0

    M_LENS = SpeosSim.Material.Create()
    M_LENS.Name = "M_LENS"
    M_LENS.OpticalPropertiesType = SpeosSim.Material.EnumOpticalPropertiesType.Volumic
    M_LENS.VOPType = SpeosSim.Material.EnumVOPType.Optic
    M_LENS.VOPIndex = 1.586
    M_LENS.VOPAbsorption = 0
    M_LENS.SOPType = SpeosSim.Material.EnumSOPType.OpticalPolished

    M_LED = SpeosSim.Material.Create()
    M_LED.Name = "M_LED"
    M_LED.OpticalPropertiesType = SpeosSim.Material.EnumOpticalPropertiesType.Volumic
    M_LED.SOPType = SpeosSim.Material.EnumSOPType.Mirror

    M_Reflector = SpeosSim.Material.Create()
    M_Reflector.Name = "M_Reflector"
    M_Reflector.OpticalPropertiesType = SpeosSim.Material.EnumOpticalPropertiesType.Volumic
    M_Reflector.SOPType = SpeosSim.Material.EnumSOPType.Mirror
    M_Reflector.SOPReflectance = 80

    # Light source
    surfaceSource = SpeosSim.SourceSurface.Create()
    surfaceSource.FluxType = SpeosSim.SourceSurface.EnumFluxType.LuminousFlux
    surfaceSource.FluxValueLuminous = 100
    surfaceSource.SpectrumType = SpeosSim.SourceSurface.EnumSpectrumType.Blackbody
    surfaceSource.SpectrumValueTemperature = 6500
    surfaceSource.IntensityType = SpeosSim.SourceSurface.EnumIntensityType.Lambertian
    surfaceSource.IntensityTotalAngle = 120
    surfaceSource.RayLength = 100
    GeoLED = GetBodyName("LED")
    surfaceSource.EmissiveFaces.Set(GeoLED.Faces[3])

    # Sensor
    IntensitySensor = SpeosSim.SensorIntensity.Create()
    coord = GetRootPart().CoordinateSystems[0]
    IntensitySensor.OriginPoint.Set(coord)
    IntensitySensor.XDirection.Set(coord.Axes[2])
    IntensitySensor.YDirection.Set(coord.Axes[0])
    IntensitySensor.XStart = -25
    IntensitySensor.XEnd = 25
    IntensitySensor.YStart = -25
    IntensitySensor.YEnd = 25
    IntensitySensor.XResolution = 2.5
    IntensitySensor.YResolution = 2.5

    # Simulation
    directSimulation = SpeosSim.SimulationDirect.Create()
    directSimulation.Geometries.SelectAll()
    directSimulation.Sources.SelectAll()
    directSimulation.Sensors.SelectAll()
    directSimulation.NbRays = 5000000

def run_simulation_step(reflector_file_path):
    wait_count = 0
    while not os.path.exists(reflector_file_path):
        if wait_count % 10 == 0:
            print("⏳ Waiting for reflector file")
        time.sleep(0.25)
        wait_count += 1

    Reflector = GetCompoenetName("Reflector")
    if Reflector:
        Reflector.Delete()

    SpeosAsm.CADUpdate.Import(reflector_file_path)
    AssignMaterial()
    directSimulation.Geometries.SelectAll()
    directSimulation.Compute()


# ✅ 해석 셋업 조건 확인 (1번째 줄이 1일 때만 실행)
with open(SpeosControl, "r") as f:
    with open(SpeosControl, "r") as f:
        flag = f.read().strip()

if flag == "0":
    OpticalAnalysisSetup(StructureData, ReflectorData)
    print("✅ Simulation setup executed.")
else:
    print("❌ Setup skipped.")

M_HSG = SpeosSim.Material.Find("M_HSG") 
M_LENS = SpeosSim.Material.Find("M_LENS") 
M_LED = SpeosSim.Material.Find("M_LED") 
M_Reflector = SpeosSim.Material.Find("M_Reflector") 

target_body_material = {
    "HSG":       ("body", M_HSG),
    "LENS":      ("body", M_LENS),
    "LED":       ("body", M_LED),
    "Reflector": ("component", M_Reflector)
}

AssignMaterial()

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

    time.sleep(0.25)
