from System.IO import Path
import time

# Global simulation objects
MaterialReflector = None
directSimulation = None

def OpticalAnalysisSetup(StructureData, ReflectorData):
    global MaterialReflector, directSimulation

    SpeosAsm.CADUpdate.Import(StructureData)
    SpeosAsm.CADUpdate.Import(ReflectorData)

    # Geometries
    GeoHSG = GetRootPart().Components[0].GetBodies()[0]
    GeoLENS = GetRootPart().Components[0].GetBodies()[1]
    GeoLED = GetRootPart().Components[0].GetBodies()[2]
    GeoReflector = GetRootPart().Components[1].GetBodies()[0]

    # Materials
    MaterialHSG = SpeosSim.Material.Create()
    MaterialHSG.Name = "HSG"
    MaterialHSG.OpticalPropertiesType = SpeosSim.Material.EnumOpticalPropertiesType.Volumic
    MaterialHSG.SOPType = SpeosSim.Material.EnumSOPType.Mirror
    MaterialHSG.SOPReflectance = 0
    MaterialHSG.VolumeGeometries.Set(GeoHSG)

    MaterialLENS = SpeosSim.Material.Create()
    MaterialLENS.Name = "LENS"
    MaterialLENS.OpticalPropertiesType = SpeosSim.Material.EnumOpticalPropertiesType.Volumic
    MaterialLENS.VOPType = SpeosSim.Material.EnumVOPType.Optic
    MaterialLENS.VOPIndex = 1.586
    MaterialLENS.VOPAbsorption = 0
    MaterialLENS.SOPType = SpeosSim.Material.EnumSOPType.OpticalPolished
    MaterialLENS.VolumeGeometries.Set(GeoLENS)

    MaterialLED = SpeosSim.Material.Create()
    MaterialLED.Name = "LED"
    MaterialLED.OpticalPropertiesType = SpeosSim.Material.EnumOpticalPropertiesType.Volumic
    MaterialLED.SOPType = SpeosSim.Material.EnumSOPType.Mirror
    MaterialLED.VolumeGeometries.Set(GeoLED)

    MaterialReflector = SpeosSim.Material.Create()
    MaterialReflector.Name = "Reflector"
    MaterialReflector.OpticalPropertiesType = SpeosSim.Material.EnumOpticalPropertiesType.Volumic
    MaterialReflector.SOPType = SpeosSim.Material.EnumSOPType.Mirror
    MaterialReflector.SOPReflectance = 80
    MaterialReflector.VolumeGeometries.Set(GeoReflector)

    # Light source
    surfaceSource = SpeosSim.SourceSurface.Create()
    surfaceSource.FluxType = SpeosSim.SourceSurface.EnumFluxType.LuminousFlux
    surfaceSource.FluxValueLuminous = 100
    surfaceSource.SpectrumType = SpeosSim.SourceSurface.EnumSpectrumType.Blackbody
    surfaceSource.SpectrumValueTemperature = 6500
    surfaceSource.IntensityType = SpeosSim.SourceSurface.EnumIntensityType.Lambertian
    surfaceSource.IntensityTotalAngle = 120
    surfaceSource.RayLength = 100
    surfaceSource.EmissiveFaces.Set(GeoLED.Faces[3])

    # Sensor
    IntensitySensor = SpeosSim.SensorIntensity.Create()
    coord = GetRootPart().CoordinateSystems[0]
    IntensitySensor.OriginPoint.Set(coord)
    IntensitySensor.XDirection.Set(coord.Axes[0])
    IntensitySensor.YDirection.Set(coord.Axes[1])
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
    directSimulation.NbRays = 10000000


def run_simulation_step(reflector_file_path):
    global MaterialReflector, directSimulation

    # 이전 Reflector 삭제
    GetRootPart().Components[1].Delete()

    # 새로운 Reflector 임포트
    SpeosAsm.CADUpdate.Import(reflector_file_path)
    GeoReflector = GetRootPart().Components[1].GetBodies()[0]

    # 물성 재적용 및 해석
    MaterialReflector.VolumeGeometries.Set(GeoReflector)
    directSimulation.Geometries.SelectAll()
    directSimulation.Compute()


StructureData = "E:/Programing/rl_reflector_project_second/CAD/Proto3D.CATPart"
ReflectorData = "E:/Programing/rl_reflector_project_second/CAD/Reflector.stp"
SpeosControl = "E:/Programing/rl_reflector_project_second/sim/SpeosControl.txt"



# ✅ 해석 셋업 조건 확인 (1번째 줄이 1일 때만 실행)
with open(SpeosControl, "r") as f:
    with open(SpeosControl, "r") as f:
        flag = f.read().strip()

if flag == "0":
    OpticalAnalysisSetup(StructureData, ReflectorData)
    print("✅ Simulation setup executed.")
else:
    print("❌ Setup skipped.")

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

