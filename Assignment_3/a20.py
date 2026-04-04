

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split



class ConvBN(nn.Module):


    def __init__(self, in_ch, out_ch, kernel_size=3,
                 stride=1, padding=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
 

    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv_bn1 = ConvBN(in_ch, out_ch, stride=stride)
        self.relu     = nn.ReLU(inplace=True)
        self.conv_bn2 = ConvBN(out_ch, out_ch, stride=1)

        # Determine whether a projection / padding is needed
        self.downsample_needed = (stride != 1) or (in_ch != out_ch)
        self.shortcut_stride   = stride
        self.in_ch             = in_ch
        self.out_ch            = out_ch

    def _option_a_shortcut(self, x):
   
        # Sub-sample: take every `stride`-th pixel
        x_sub = x[:, :, ::self.shortcut_stride, ::self.shortcut_stride]
        # Channel padding: prepend and append zeros to double channel count
        pad_ch = self.out_ch - self.in_ch
        # Pad (pad_ch//2) on each side of the channel dimension
        half   = pad_ch // 2
        zeros  = torch.zeros_like(x_sub[:, :half])
        return torch.cat([zeros, x_sub, zeros], dim=1)

    def forward(self, x):
        residual = x

        out = self.conv_bn1(x)
        out = self.relu(out)
        out = self.conv_bn2(out)

        if self.downsample_needed:
            residual = self._option_a_shortcut(x)

        out = self.relu(out + residual)
        return out



class ResNet20(nn.Module):


    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        n = 3  

   
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(
            in_ch=16, out_ch=16, num_blocks=n, first_stride=1
        )

        self.stage2 = self._make_stage(
            in_ch=16, out_ch=32, num_blocks=n, first_stride=2
        )

        self.stage3 = self._make_stage(
            in_ch=32, out_ch=64, num_blocks=n, first_stride=2
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc              = nn.Linear(64, num_classes)

        self._initialise_weights()

    @staticmethod
    def _make_stage(in_ch, out_ch, num_blocks, first_stride):
   
        blocks = [ResidualBlock(in_ch, out_ch, stride=first_stride)]
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*blocks)

    def _initialise_weights(self):
        """Kaiming (He) normal initialisation for conv layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias,   0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.initial_conv(x)   # 3→16 channels, 32×32
        x = self.stage1(x)         # 16 channels,   32×32
        x = self.stage2(x)         # 32 channels,   16×16
        x = self.stage3(x)         # 64 channels,    8×8
        x = self.global_avg_pool(x)  # 64 channels,   1×1
        x = x.view(x.size(0), -1)   # flatten → (B, 64)
        x = self.fc(x)             # → (B, 10)
        return x




def build_dataloaders(data_root='./data', batch_size=128,
                      num_workers=2, val_split=True):
  
    MEAN = (0.4914, 0.4822, 0.4465)
    STD  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.Pad(4),                         # 4 px on each side → 40×40
        transforms.RandomCrop(32),                 # random 32×32 crop
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root=data_root, train=True,
        download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False,
        download=True, transform=test_transform
    )

    if val_split:
        train_set, val_set = random_split(
            full_train, [45000, 5000],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        train_set, val_set = full_train, None

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=128,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    val_loader   = None
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=128,
                                shuffle=False, num_workers=num_workers,
                                pin_memory=True)
    return train_loader, val_loader, test_loader




def evaluate(model, loader, device):
    """Return top-1 accuracy (%) on the given data loader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions    = model(images).argmax(dim=1)
            correct       += (predictions == labels).sum().item()
            total         += labels.size(0)
    return 100.0 * correct / total


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Single training epoch; returns average loss."""
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)




def main():
    # ── Device ────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[ResNet-20]  Running on: {device}")

  
    BATCH_SIZE  = 128
    WEIGHT_DECAY= 1e-4
    MOMENTUM    = 0.9
    BASE_LR     = 0.1

    TOTAL_EPOCHS   = 182
    LR_MILESTONES  = [91, 136]
    LR_GAMMA       = 0.1

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=BATCH_SIZE
    )

    model = ResNet20(num_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[ResNet-20]  Parameters: {total_params:,}  "
          f"(paper reports ~0.27 M)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=BASE_LR,
                          momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY,
                          nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=LR_MILESTONES, gamma=LR_GAMMA
    )

  
    best_val_acc = 0.0
    for epoch in range(1, TOTAL_EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader,
                                   criterion, optimizer, device)
        scheduler.step()

        if epoch % 10 == 0 or epoch == TOTAL_EPOCHS:
            val_acc  = evaluate(model, val_loader,  device)
            test_acc = evaluate(model, test_loader, device)
            lr_now   = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch:3d}/{TOTAL_EPOCHS}]  "
                  f"Loss: {avg_loss:.4f}  "
                  f"Val: {val_acc:.2f}%  "
                  f"Test: {test_acc:.2f}%  "
                  f"LR: {lr_now:.5f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'resnet20_best.pth')


    model.load_state_dict(torch.load('resnet20_best.pth'))
    final_acc = evaluate(model, test_loader, device)
    print(f"\n[ResNet-20]  Final Test Accuracy : {final_acc:.2f}%")
    print(f"             Paper reports       : ~91.25%  (best run)")


if __name__ == '__main__':
    main()