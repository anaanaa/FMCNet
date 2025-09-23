import torch
import torch.nn as nn


class Boundary_loss(nn.Module):
    def __init__(self, boundary_kernel_size=3):
        super(Boundary_loss, self).__init__()

        # 平均池化层，计算平滑后的边缘
        self.pooling_layer = nn.AvgPool3d(
            (boundary_kernel_size, boundary_kernel_size, 1),
            stride=1,
            padding=(int((boundary_kernel_size - 1) / 2),
                     int((boundary_kernel_size - 1) / 2),
                     0)
        ).cuda()

        self.loss_boundary = nn.L1Loss(reduction='mean')

    def forward(self, masks, seg):
        with torch.no_grad():
            if masks.ndim != seg.ndim:
                seg = seg.view((seg.shape[0], 1, *seg.shape[1:]))
            if masks.shape == seg.shape:
                y_onehot = seg
            else:
                # 对 gt 进行独热编码
                y_onehot = torch.zeros(masks.shape, device=masks.device)
                y_onehot.scatter_(1, seg.long(), 1)
        seg = y_onehot

        if seg.sum() > 0:
            seg = (seg > 0).float()
            masks = (masks > 0).float()
            seg_edge = abs(seg - self.pooling_layer(seg))  # Calculate edge for seg
            #mask_probs = torch.softmax(masks, dim=1)
            mask_edge = abs(masks - self.pooling_layer(masks))  # Calculate edge for mask
            # Compute the loss
            loss_distance = self.loss_boundary(mask_edge, seg_edge) * 10
            #print(mask_edge, seg_edge)
        else:
            loss_distance = torch.tensor(0).cuda()

        return loss_distance


# 测试代码
if __name__ == '__main__':
    # 假设我们有一个批量大小为1，4个类别，尺寸为10x10x10的3D张量
    batch_size = 1
    num_classes = 3  # 假设有4个类别
    depth, height, width = 4, 4, 4

    # 创建一个假的seg张量和masks张量
    # seg是一个包含0到3类别的张量，代表多类别分割
    seg = torch.randint(0, num_classes, (batch_size, depth, height, width)).cuda()

    # masks是一个假的概率分布张量，形状为 [batch_size, num_classes, depth, height, width]
    masks = torch.randn(batch_size, num_classes, depth, height, width).cuda()

    # 实例化 boundary_loss 类
    criterion = Boundary_loss(boundary_kernel_size=3)

    print('seg',seg.shape,'masks',masks.shape)

    # 前向传播，计算边界损失
    loss = criterion(masks, seg)

    print(f"Boundary Loss: {loss.item()}")
