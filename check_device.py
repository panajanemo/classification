#!/usr/bin/env python3
"""
检测Mac设备加速能力
"""

import torch
import platform

def check_devices():
    """检测可用的计算设备"""
    print("=== Mac设备加速能力检测 ===\n")
    
    print(f"系统信息: {platform.system()} {platform.machine()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"PyTorch版本: {torch.__version__}")
    print()
    
    print("可用设备:")
    
    # 检测CPU
    print("✅ CPU: 始终可用")
    
    # 检测CUDA (不太可能在Mac上)
    if torch.cuda.is_available():
        print("✅ CUDA: 可用")
        print(f"   设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   设备{i}: {torch.cuda.get_device_name(i)}")
    else:
        print("❌ CUDA: 不可用")
    
    # 检测MPS (Metal Performance Shaders)
    if hasattr(torch.backends, 'mps'):
        if torch.backends.mps.is_available():
            print("✅ MPS (Metal): 可用 - 推荐用于Mac加速")
            
            # 测试MPS设备
            try:
                mps_device = torch.device("mps")
                test_tensor = torch.randn(100, 100).to(mps_device)
                print("   MPS测试: 通过")
            except Exception as e:
                print(f"   MPS测试: 失败 ({e})")
        else:
            print("❌ MPS (Metal): 不可用")
    else:
        print("❌ MPS (Metal): PyTorch版本不支持")
    
    print()
    
    # 自动选择推荐设备
    def get_recommended_device():
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    recommended = get_recommended_device()
    print(f"推荐设备: {recommended}")
    
    if recommended == "mps":
        print("\n🚀 你的Mac支持MPS加速！")
        print("建议设置环境变量: SEMANTIC_DEVICE=mps")
        print("或者在.env文件中设置: SEMANTIC_DEVICE=mps")
        print("这将显著提升BERT模型的推理速度")
    elif recommended == "cpu":
        print("\n⚠️  仅可使用CPU，推理速度较慢")
        print("如果你的Mac支持Metal，请升级PyTorch版本")
    
    print()
    return recommended

def performance_test():
    """简单的性能测试"""
    print("=== 设备性能测试 ===\n")
    
    devices_to_test = ["cpu"]
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices_to_test.append("mps")
    
    if torch.cuda.is_available():
        devices_to_test.append("cuda")
    
    import time
    
    for device_name in devices_to_test:
        print(f"测试设备: {device_name}")
        
        try:
            device = torch.device(device_name)
            
            # 创建测试张量
            start_time = time.time()
            x = torch.randn(1000, 768).to(device)
            y = torch.randn(768, 512).to(device)
            
            # 执行矩阵乘法
            for _ in range(100):
                z = torch.mm(x, y)
            
            # 确保计算完成
            if device_name == "cuda":
                torch.cuda.synchronize()
            elif device_name == "mps":
                torch.mps.synchronize()
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            print(f"   100次矩阵乘法耗时: {elapsed:.3f}秒")
            
        except Exception as e:
            print(f"   测试失败: {e}")
        
        print()

def main():
    """主函数"""
    recommended_device = check_devices()
    
    print("=" * 50)
    performance_test()
    
    print("=" * 50)
    print("配置建议:")
    print(f"export SEMANTIC_DEVICE={recommended_device}")
    print(f"或在.env文件中添加: SEMANTIC_DEVICE={recommended_device}")

if __name__ == "__main__":
    main()