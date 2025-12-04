import torch
import torch.nn as nn
from models import ResNeXt29_8x64d
from utils import get_cifar10_dataloaders


def test_model(model_path=None):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载
    _, testloader, classes = get_cifar10_dataloaders(batch_size=100)

    # 模型初始化
    model = ResNeXt29_8x64d().to(device)

    # 加载预训练权重
    if model_path and torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    elif model_path:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print("请提供模型路径")
        return

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 每个类别的准确率
            c = (predicted == targets).squeeze()
            for i in range(targets.size(0)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # 总体准确率
    print(f'总体准确率: {100. * correct / total:.2f}%')
    print(f'正确数量: {correct}/{total}')

    # 每个类别的准确率
    print('\n每个类别的准确率:')
    for i in range(10):
        if class_total[i] > 0:
            print(f'{classes[i]:10s}: {100 * class_correct[i] / class_total[i]:.2f}%')
        else:
            print(f'{classes[i]:10s}: N/A')


if __name__ == '__main__':
    # 测试训练好的模型，将路径替换为实际模型路径
    test_model('resnext29_final.pth')