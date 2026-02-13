import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# --- 1. 设备配置 (利用 GPU) ---
# 检查是否有可用的 CUDA GPU，否则使用 CPU
# print(torch.__version__)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")


# --- 2. 数据预处理与加载 (对应你的 data_preprocess.py) ---
# PyTorch 的 torchvision 提供了强大的数据预处理和加载工具
# 定义数据转换：转换为 Tensor 并进行归一化
# CIFAR-10 的均值和标准差（经验值）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 使用 torchvision 自带的数据集加载器
# 这比手动读取 pickle 文件更方便、高效
data_dir = './data' # 数据将下载并存储到这个目录
train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

# 创建 DataLoader
# DataLoader 负责批量 (batching)、打乱 (shuffling) 和多线程加载数据
batch_size = 128 # 可以根据你的 GPU 显存调整
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# --- 3. 构建模型 (对应你的 model.py 和 ConvModulePack.py) ---
# 在 PyTorch 中，所有模型都继承自 nn.Module
# 你的 Model 类和 add 方法 -> nn.Sequential 或在 __init__ 中定义层
class VGG_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_CIFAR10, self).__init__()
        
        # 特征提取层 (对应你的 Conv2D, ReLU, MaxPooling2D)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # 对应 Conv2D(in_channels=3, out_channels=64)
            nn.ReLU(inplace=True),                      # 对应 ReLU()
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 对应 MaxPooling2D()

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 分类器层 (对应你的 Flatten 和 Dense)
        self.classifier = nn.Sequential(
            # nn.Flatten() 也可以，但这里手动计算更清晰
            nn.Linear(256 * 4 * 4, 1024),               # 对应 Dense(input_dim=..., output_dim=...)
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),                            # 添加 Dropout 防止过拟合
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    # forward 方法定义了数据如何流经网络
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # 对应 Flatten()
        x = self.classifier(x)
        return x

# 实例化模型并将其移动到 GPU
model = VGG_CIFAR10().to(device)


# --- 4. 定义损失函数和优化器 (对应你的 optimization.py) ---
# SoftmaxCrossEntropy -> nn.CrossEntropyLoss (PyTorch 的这个损失函数内置了 Softmax)
criterion = nn.CrossEntropyLoss()

# SGD -> optim.SGD。Adam 通常是更好的默认选择。
optimizer = optim.Adam(model.parameters(), lr=0.001)


# --- 5. 训练模型 (对应你的 model.py 中的 fit 方法) ---
def train(epochs):
    for epoch in range(epochs):
        start_time = time.time()
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            # 获取输入数据；data 是一个 [inputs, labels] 的列表
            inputs, labels = data
            # 将数据移动到 GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # 1. 梯度清零 (非常重要！)
            optimizer.zero_grad()

            # 2. 前向传播
            outputs = model(inputs)
            
            # 3. 计算损失
            loss = criterion(outputs, labels)
            
            # 4. 反向传播
            loss.backward()
            
            # 5. 更新权重
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:  # 每 200 个 mini-batches 打印一次信息
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0
        
        # 每个 epoch 结束后，在测试集上评估模型
        validate(epoch)
        
        end_time = time.time()
        print(f"Epoch {epoch+1} finished in {end_time - start_time:.2f} seconds.")

    print('Finished Training')


# --- 6. 评估模型 ---
def validate(epoch):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估时，不需要计算梯度
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f'Accuracy on the test set after epoch {epoch + 1}: {accuracy:.2f} %')


if __name__ == '__main__':
    num_epochs = 20 # 增加 epochs 以获得更好的性能
    train(num_epochs)
    
    # 保存模型 (对应你的 save_weights)
    # PyTorch 通常保存 state_dict (状态字典)，只包含模型的权重和偏置
    save_path = './cifar_vgg_net.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # 加载模型示例
    # new_model = VGG_CIFAR10().to(device)
    # new_model.load_state_dict(torch.load(save_path))
    # print("Model loaded.")