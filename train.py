import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
import time
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import datetime

class PreActResNet(nn.Module):
    """Pre-activation ResNet for CIFAR-10"""

    def __init__(self, num_blocks=[9, 9, 9], num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)

        self.bn_final = nn.BatchNorm2d(256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.relu(self.bn_final(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class PreActBlock(nn.Module):
    """Pre-activation 残差块"""

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(torch.relu(self.bn2(out)))
        return out + shortcut


def cutmix_data(x, y, alpha=1.0):
    """CutMix数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    # 生成剪裁区域
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # 调整lambda值
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x, y, y[index], lam


def mixup_data(x, y, alpha=1.0):
    """MixUp数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def rand_bbox(size, lam):
    """生成随机边界框"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 随机位置
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        if target.dim() == 2:  # Mixup情况 - target是one-hot格式
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)
            nll_loss = - (logprobs * target).sum(dim=-1)
            smooth_loss = - logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:  # 普通情况
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()


def get_advanced_transforms():
    """SOTA级别的数据增强"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    return transform_train, transform_test


def train_sota_optimized():

    # # 🚀 硬件设置
    # gpu_ids = [3, 4]
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

    # 🎯 优化后的超参数
    batch_size = 256
    epochs = 200
    lr = 0.05 # 需要根据不同的优化器进行调整

    device = torch.device('cuda')
    print(f"🎯 SOTA训练")
    print(f"📊 优化超参数: batch_size={batch_size}, epochs={epochs}, lr={lr}")

    # 初始化TensorBoard
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f'runs/cifar10_sota_optimized_{timestamp}')
    print("📊 TensorBoard已启动，使用命令: tensorboard --logdir=runs/")

    # 数据加载
    transform_train, transform_test = get_advanced_transforms()

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=16, pin_memory=True, drop_last=True)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=16, pin_memory=True)

    # 模型
    model = PreActResNet(num_blocks=[9, 9, 9]).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"🔄 使用 {torch.cuda.device_count()} 个GPU进行数据并行")

    # SGD with Nesterov优化器
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=lr,
    #     momentum=0.9,
    #     weight_decay=5e-4,
    #     nesterov=True
    # )
    # 优化器Adagrad
    optimizer = optim.Adagrad(
        model.parameters(),
        lr=lr,
        lr_decay=1e-6,  # Adagrad特有的学习率衰减，防止学习率过快下降
        weight_decay=5e-4,  # 增加权重衰减防止过拟合
        initial_accumulator_value=0.1,  # 初始化梯度累积值
    )
    # 优化器 SGD with momentum
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=lr,
    #     momentum=0.9,
    #     weight_decay=5e-4,
    #     nesterov=True
    # )

    # 优化器adamw
    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr=lr,
    #     weight_decay=0.01
    # )

    # 优化器adam
    # try:
    #     from torch_optimizer import RAdam
    #     optimizer = RAdam(
    #         model.parameters(),
    #         lr=lr,
    #         weight_decay=0.01
    #     )
    #     print("✅ 使用RAdam优化器")
    # except ImportError:
    #     print("⚠️  RAdam未安装，使用Adam代替")
    #     optimizer = optim.Adam(
    #         model.parameters(),
    #         lr=lr,
    #         weight_decay=0.05
    #
    # 学习率调度策略
    # 针对Adagrad的预热+余弦退火
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

    # 预热阶段：前10个epoch线性增加学习率
    warmup_epochs = 10
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,  # 从lr的1%开始
        end_factor=1.0,
        total_iters=warmup_epochs * len(trainloader)
    )

    # 主训练阶段：余弦退火
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(epochs - warmup_epochs) * len(trainloader),
        eta_min=1e-5
    )

    # 组合调度器
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs * len(trainloader)]
    )
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer,
    #     milestones=[30, 60, 90, 130, 160],  # 提前或在这些点衰减
    #     gamma=0.1  # 每次衰减为原来的0.1倍
    # )
    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(trainloader))

    # from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    # scheduler = CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=50,  # 初始重启周期
    #     T_mult=2,  # 每次重启周期加倍
    #     eta_min=1e-6  # 最小学习率
    # )
    # OneCycleLR配合RAdam效果很好
    # from torch.optim.lr_scheduler import OneCycleLR

    # scheduler = OneCycleLR(
    #     optimizer,
    #     max_lr=0.005,  # 最大学习率可以设高一点
    #     epochs=epochs,
    #     steps_per_epoch=len(trainloader),
    #     pct_start=0.1,  # RAdam需要较短的warmup（10%）
    #     anneal_strategy='cos',
    #     cycle_momentum=False,  # RAdam不需要momentum cycling
    #     div_factor=10.0,  # 初始学习率 = max_lr/10
    #     final_div_factor=1000.0  # 最终学习率 = max_lr/1000
    # )
    # 损失函数 - 现在已定义
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if hasattr(torch.cuda.amp, 'GradScaler') else None
    use_amp = scaler is not None
    print(f"🔧 混合精度训练: {'启用' if use_amp else '未启用'}")

    print("🚀 开始优化版SOTA训练...")
    start_time = time.time()
    best_acc = 0
    global_step = 0

    # 创建checkpoints目录
    os.makedirs('./checkpoints', exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, desc=f'Epoch {epoch + 1}/{epochs}')):
            inputs, targets = inputs.to(device), targets.to(device)

            # 数据增强
            use_mixup = random.random() < 0.5
            use_cutmix = random.random() < 0.5 and not use_mixup

            if use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)
            elif use_cutmix:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, alpha=1.0)
            else:
                targets_a = targets
                targets_b = targets
                lam = 1.0

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)

                    if use_mixup or use_cutmix:
                        targets_a_onehot = torch.zeros(outputs.size()).to(device)
                        targets_a_onehot.scatter_(1, targets_a.unsqueeze(1), 1)
                        targets_b_onehot = torch.zeros(outputs.size()).to(device)
                        targets_b_onehot.scatter_(1, targets_b.unsqueeze(1), 1)

                        loss = lam * criterion(outputs, targets_a_onehot) + (1 - lam) * criterion(outputs,
                                                                                                  targets_b_onehot)
                    else:
                        loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                if use_mixup or use_cutmix:
                    targets_a_onehot = torch.zeros(outputs.size()).to(device)
                    targets_a_onehot.scatter_(1, targets_a.unsqueeze(1), 1)
                    targets_b_onehot = torch.zeros(outputs.size()).to(device)
                    targets_b_onehot.scatter_(1, targets_b.unsqueeze(1), 1)

                    loss = lam * criterion(outputs, targets_a_onehot) + (1 - lam) * criterion(outputs, targets_b_onehot)
                else:
                    loss = criterion(outputs, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            running_loss += loss.item()

            # TensorBoard记录 - 每个batch
            writer.add_scalar('Training/Loss_batch', loss.item(), global_step)
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)

            # 准确率计算
            _, predicted = outputs.max(1)
            total += targets.size(0)

            if use_mixup or use_cutmix:
                correct += (lam * predicted.eq(targets_a).sum().item() +
                            (1 - lam) * predicted.eq(targets_b).sum().item())
            else:
                correct += predicted.eq(targets).sum().item()

            batch_acc = 100. * correct / total
            writer.add_scalar('Training/Accuracy_batch', batch_acc, global_step)

            global_step += 1

        # 每个epoch结束后评估
        train_acc = 100. * correct / total
        test_acc = evaluate_full(model, testloader, device)

        # TensorBoard记录 - 每个epoch
        writer.add_scalar('Training/Accuracy_epoch', train_acc, epoch)
        writer.add_scalar('Testing/Accuracy', test_acc, epoch)
        writer.add_scalar('Training/Loss_epoch', running_loss / len(trainloader), epoch)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint_path = f'./checkpoints/best_model_sota_optimized_{timestamp}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
                'epoch': epoch
            }, checkpoint_path)
            print(f"💾 保存最佳模型到: {checkpoint_path}")

        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - start_time

        print(f'Epoch [{epoch + 1:3d}/{epochs}] | LR: {current_lr:.4f} | '
              f'Train: {train_acc:5.2f}% | Test: {test_acc:5.2f}% | '
              f'Best: {best_acc:5.2f}% | Time: {epoch_time / 60:5.1f}m')

    final_time = time.time() - start_time
    print(f'\n🏆 训练完成! 最佳准确率: {best_acc:.2f}%')
    print(f'⏱️ 总时间: {final_time / 60:.1f}分钟')

    writer.close()
    return best_acc


def evaluate_full(model, testloader, device):
    """完整评估"""
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
    return 100. * correct / total


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    accuracy = train_sota_optimized()