import os
import torch
import subprocess
import psutil


def optimize_system(gpu_ids=[3, 4, 5, 6]):
    """系统级优化脚本"""

    print("=" * 60)
    print("🚀 系统极速优化启动")
    print(f"🎯 指定GPU: {gpu_ids}")
    print("=" * 60)

    # 1. 设置CUDA环境变量 - 指定GPU
    print("\n1. 🔧 设置CUDA环境变量...")
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    os.environ['CUDA_CACHE_PATH'] = '/tmp/cuda-cache'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    # 2. PyTorch性能优化
    print("2. ⚡ PyTorch性能优化...")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 3. 检查指定GPU状态
    print("3. 🔍 检查指定GPU状态...")
    if torch.cuda.is_available():
        visible_gpus = list(range(torch.cuda.device_count()))
        print(f"   ✅ 可见的GPU设备: {visible_gpus}")

        # 检查实际可用的GPU
        actual_gpus = []
        for i in visible_gpus:
            try:
                torch.cuda.set_device(i)
                props = torch.cuda.get_device_properties(i)
                actual_physical_id = gpu_ids[i] if i < len(gpu_ids) else i
                actual_gpus.append(actual_physical_id)
                memory = props.total_memory / 1024 ** 3
                print(f"   📊 GPU {actual_physical_id}(虚拟{i}): {props.name}")
                print(f"     内存: {memory:.1f}GB")
                print(f"     Compute Capability: {props.major}.{props.minor}")
            except Exception as e:
                print(f"   ❌ GPU {gpu_ids[i]} 不可用: {e}")

        if len(actual_gpus) < len(gpu_ids):
            print(f"   ⚠️  警告: 只有 {len(actual_gpus)}/{len(gpu_ids)} 个GPU可用")

        # 清理所有GPU缓存
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()

        print(f"   🎯 实际使用的GPU: {actual_gpus}")
    else:
        print("   ❌ 未检测到GPU，将使用CPU（性能较差）")

    # 4. 检查系统资源
    print("4. 💻 检查系统资源...")
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    print(f"   CPU核心数: {cpu_count}")
    print(f"   内存: {memory.total / 1024 ** 3:.1f}GB, 可用: {memory.available / 1024 ** 3:.1f}GB")

    # 5. 设置进程优先级（Linux）
    if os.name == 'posix':
        print("5. 🎯 设置进程优先级...")
        try:
            os.nice(-10)
            print("   ✅ 进程优先级已提升")
        except:
            print("   ⚠️  无法提升进程优先级（需要sudo权限）")

    # 6. 验证优化结果
    print("6. ✅ 验证优化结果...")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    print(f"   可见GPU数量: {torch.cuda.device_count()}")
    print(f"   cuDNN基准模式: {torch.backends.cudnn.benchmark}")
    print(f"   TF32矩阵乘法: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"   TF32卷积: {torch.backends.cudnn.allow_tf32}")

    print("\n" + "=" * 60)
    print("🎉 系统优化完成！现在可以运行极速训练。")
    print("💡 运行命令: python train_30min.py")
    print("=" * 60)


def check_training_readiness():
    """检查训练准备状态"""
    print("\n🔍 训练准备状态检查:")

    # 检查必要的文件
    required_files = ['train_30min.py', 'models/__init__.py', 'utils/data_loader.py']
    missing_files = []

    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print(f"   ❌ 缺少必要文件: {missing_files}")
        return False
    else:
        print("   ✅ 所有必要文件都存在")

    # 检查GPU内存
    if torch.cuda.is_available():
        total_free_memory = 0
        for i in range(torch.cuda.device_count()):
            free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            free_gb = free_memory / 1024 ** 3
            total_free_memory += free_gb

            if free_gb < 4:
                print(f"   ⚠️  GPU {i} 空闲内存不足: {free_gb:.1f}GB (建议≥4GB)")
            else:
                print(f"   ✅ GPU {i} 空闲内存充足: {free_gb:.1f}GB")

        print(f"   📊 总可用GPU内存: {total_free_memory:.1f}GB")

        # 根据总内存推荐批大小
        if total_free_memory >= 60:
            recommended_batch = 4096
        elif total_free_memory >= 40:
            recommended_batch = 2048
        elif total_free_memory >= 20:
            recommended_batch = 1024
        else:
            recommended_batch = 512

        print(f"   💡 推荐批大小: {recommended_batch}")

    return True


if __name__ == '__main__':
    gpu_ids = [3, 4, 5, 6]
    optimize_system(gpu_ids)

    # 检查训练准备状态
    if check_training_readiness():
        print("\n🎯 系统已准备好进行极速训练！")
    else:
        print("\n❌ 请先解决上述问题再开始训练。")