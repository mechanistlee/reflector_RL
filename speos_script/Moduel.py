# 각 물성에 해당하는 Body 를 찾아서 할당하는 스크립트
M_HSG = SpeosSim.Material.Find("M_HSG")  # HSG 물성 객체 가져오기
M_LENS = SpeosSim.Material.Find("M_LENS")  # LENS 물성 객체 가져오기
M_LED = SpeosSim.Material.Find("M_LED")  # LED 물성 객체 가져오기
M_Reflector = SpeosSim.Material.Find("M_Reflector")  # Reflector 물성 객체 가져오기

target_body_material = {
    "HSG": M_HSG,
    "LENS": M_LENS,
    "LED": M_LED,
    "Reflector": M_Reflector
}

# 바디 이름으로 찾기
def get_body_by_name(target_name):
    for comp in GetRootPart().Components:
        for body in comp.GetBodies():
            if body.Name == target_name:
                return body
    return None

# 컴포넌트 이름으로 찾기
def get_component_by_name(target_name):
    for comp in GetRootPart().Components:
        if comp.GetName() == target_name:
            return comp
    return None

# 각 물성에 해당하는 Body 를 찾아서 할당하는 스크립트
def AssignMaterial():
    found_names = []
    for name, (target_type, mat) in target_body_material.items():
        target_obj = None
        if target_type == "body":
            target_obj = get_body_by_name(name)
        elif target_type == "component":
            target_obj = get_component_by_name(name)
        if target_obj:
            if mat:
                mat.VolumeGeometries.Set(target_obj)
                found_names.append(name)
                print("✅ 물성 '{}' 에 {} '{}' 할당 완료".format(mat.Name, target_type, name))
            else:
                print("⚠️  {} '{}' 에 할당할 물성이 없습니다 (mat is None)".format(target_type, name))
        else:
            print("⚠️  {} '{}' 를 찾을 수 없습니다.".format(target_type, name))
    if not found_names:
        print("❌ 매칭된 Body / Component 가 없습니다.")
    else:
        print("✅ 최종 할당된 대상들:", found_names)

