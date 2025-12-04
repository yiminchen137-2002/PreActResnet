import torch
import torch.nn as nn
from models import ResNeXt29_8x64d
from utils import get_cifar10_dataloaders
import os


def test_checkpoints():
    """测试保存的检查点"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载
    _, testloader, classes = get_cifar10_dataloaders(batch_size=100)

    # 模型初始化
    model = ResNeXt29_8x64d().to(device)

    # 测试不同检查点
    checkpoints = [
        ('最佳模型', 'checkpoints/model_best.pth'),
        ('最新模型', 'checkpoints/model_last.pth')
    ]

    for checkpoint_name, checkpoint_path in checkpoints:
        if os.path.exists(checkpoint_path):
            print(f"\n🔍 测试 {checkpoint_name}: {checkpoint_path}")

            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

            # 测试准确率
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in testloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            accuracy = 100. * correct / total
            print(f"📊 {checkpoint_name}测试准确率: {accuracy:.2f}%")
            print(f"📝 训练信息: Epoch {checkpoint['epoch']}, "
                  f"训练准确率: {checkpoint.get('train_acc', 'N/A'):.2f}%")
        else:
            print(f"❌ 检查点不存在: {checkpoint_path}")


if __name__ == '__main__':
    test_checkpoints()