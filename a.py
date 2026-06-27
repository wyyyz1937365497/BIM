import ifcopenshell
import ifcopenshell.geom

try:
    # 尝试初始化几何设置
    settings = ifcopenshell.geom.settings()
    # 尝试创建一个 geometry iterator (不需要实际文件也能测试内核是否加载)
    print("✅ 成功！ifcopenshell.geom 几何内核已正常加载。")
except Exception as e:
    print(f"❌ 失败：{e}")