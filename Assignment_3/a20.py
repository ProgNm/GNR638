"""
ResNet-20 on CIFAR-10
=====================
Paper : He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
        Section 4.2 — CIFAR-10 experiments

Architecture summary (from the paper):
  * n = 3  →  total weighted layers = 6n + 2 = 20
  * Stage 0  : one 3×3 conv, 16 filters, 32×32 output
  * Stage 1  : 2n = 6 layers (3 residual blocks), 16 filters, 32×32
  * Stage 2  : 2n = 6 layers (3 residual blocks), 32 filters, 16×16
  * Stage 3  : 2n = 6 layers (3 residual blocks), 64 filters,  8×8
  * Head     : global average pool → 10-way FC → softmax
  * Shortcuts: identity option-A (zero-pad channels, stride-2 sub-sample)
               — adds ZERO extra parameters
  * BN applied right after every convolution, before nonlinearity

Training hyper-parameters (reproduced exactly from paper):
  * Optimizer : SGD, momentum = 0.9, weight decay = 1e-4, no dropout
  * Batch size : 128
  * Schedule  : lr starts at 0.1; divided by 10 at iterations 32k and 48k;
                training terminates at 64k iterations
                (determined on a 45k/5k train/val split — we mimic with
                 epoch equivalents: 64k iters / (45k/128) ≈ 182 epochs,
                 milestones at ~91 and ~136 epochs)
  * Data aug  : 4 px zero-padding on each side, random 32×32 crop,
                random horizontal flip; test: original 32×32 image only
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


# ───────────────────────────────────────────────────────────────
# 1.  Basic building-block components
# ───────────────────────────────────────────────────────────────

class ConvBN(nn.Module):
    """
    A single Conv2d followed immediately by BatchNorm2d.
    Bias is omitted because the subsequent BN re-centres the output.
    Activation is NOT included here so that the residual addition can
    happen before the final ReLU (see Figure 2 in the paper).
    """

    def __init__(self, in_ch, out_ch, kernel_size=3,
                 stride=1, padding=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


# ───────────────────────────────────────────────────────────────
# 2.  Residual block  (Figure 2 of the paper)
# ───────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Two-layer residual block:
        y = ReLU( F(x) + shortcut(x) )
    where F(x) = BN(Conv(ReLU(BN(Conv(x)))))

    Shortcut is option-A from the paper:
        - Same dimensions  →  plain identity (x returned as-is)
        - Dimensions differ →  x is sub-sampled with stride then
                               zero-padded along the channel axis
                               (no learnable parameters introduced)
    """

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
        """
        Identity shortcut option A:
        Spatially sub-sample with stride, then zero-pad extra channels.
        This introduces NO extra parameters (paper Section 3.3 option A).
        """
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


# ───────────────────────────────────────────────────────────────
# 3.  Full ResNet-20 model
# ───────────────────────────────────────────────────────────────

class ResNet20(nn.Module):
    """
    ResNet with n=3  →  6*3+2 = 20 weighted layers.

    Layer layout (Table in Section 4.2):
      output 32×32 : conv(16)         — 1 layer
      output 32×32 : [3×3, 16] × 2n  — stage 1  (n=3 blocks)
      output 16×16 : [3×3, 32] × 2n  — stage 2  (n=3 blocks)
      output  8×8  : [3×3, 64] × 2n  — stage 3  (n=3 blocks)
      output  1×1  : global avg pool
      output 10    : fully-connected
    """

    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        n = 3  # paper uses n=3 for 20-layer network

        # ── Initial convolution ──────────────────────────────
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # ── Stage 1: feature map 32×32, 16 filters ───────────
        # First block: no spatial downsampling, no channel change
        self.stage1 = self._make_stage(
            in_ch=16, out_ch=16, num_blocks=n, first_stride=1
        )

        # ── Stage 2: feature map 16×16, 32 filters ───────────
        # First block: stride=2 halves spatial dims, doubles channels
        self.stage2 = self._make_stage(
            in_ch=16, out_ch=32, num_blocks=n, first_stride=2
        )

        # ── Stage 3: feature map 8×8, 64 filters ─────────────
        self.stage3 = self._make_stage(
            in_ch=32, out_ch=64, num_blocks=n, first_stride=2
        )

        # ── Classification head ───────────────────────────────
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc              = nn.Linear(64, num_classes)

        # ── Weight initialisation (He et al., ICCV 2015) ─────
        self._initialise_weights()

    @staticmethod
    def _make_stage(in_ch, out_ch, num_blocks, first_stride):
        """
        Build one stage as a Sequential of ResidualBlocks.
        Only the FIRST block may have stride=2 (to halve the feature map).
        All subsequent blocks keep stride=1 and dimensions unchanged.
        """
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


# ───────────────────────────────────────────────────────────────
# 4.  Data loading  (paper Section 4.2 augmentation)
# ───────────────────────────────────────────────────────────────

def build_dataloaders(data_root='./data', batch_size=128,
                      num_workers=2, val_split=True):
    """
    Training transform  : pad 4 px → random 32×32 crop → random H-flip
                          → normalise with CIFAR-10 channel stats
    Test transform      : normalise only (single 32×32 view, as in paper)
    """
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

    # 45k / 5k split to mirror how the paper determined the schedule
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


# ───────────────────────────────────────────────────────────────
# 5.  Training utilities
# ───────────────────────────────────────────────────────────────

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


# ───────────────────────────────────────────────────────────────
# 6.  Main training loop
# ───────────────────────────────────────────────────────────────

def main():
    # ── Device ────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[ResNet-20]  Running on: {device}")

    # ── Hyper-parameters (from paper Section 4.2) ─────────────
    BATCH_SIZE  = 128
    WEIGHT_DECAY= 1e-4
    MOMENTUM    = 0.9
    BASE_LR     = 0.1
    # 64k iterations with batch=128 over 45k training samples
    # ≈ 182 epochs; milestones at 32k→~91 ep and 48k→~136 ep
    TOTAL_EPOCHS   = 182
    LR_MILESTONES  = [91, 136]
    LR_GAMMA       = 0.1

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=BATCH_SIZE
    )

    # ── Model ─────────────────────────────────────────────────
    model = ResNet20(num_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[ResNet-20]  Parameters: {total_params:,}  "
          f"(paper reports ~0.27 M)")

    # ── Loss, optimiser, scheduler ────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=BASE_LR,
                          momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY,
                          nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=LR_MILESTONES, gamma=LR_GAMMA
    )

    # ── Training loop ─────────────────────────────────────────
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

    # ── Final test accuracy ───────────────────────────────────
    model.load_state_dict(torch.load('resnet20_best.pth'))
    final_acc = evaluate(model, test_loader, device)
    print(f"\n[ResNet-20]  Final Test Accuracy : {final_acc:.2f}%")
    print(f"             Paper reports       : ~91.25%  (best run)")


if __name__ == '__main__':
    main()