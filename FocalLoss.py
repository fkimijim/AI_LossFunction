class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def forward(self, inputs, targets):
        if self.from_logits:
            inputs = torch.sigmoid(inputs)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_loss

        return focal_loss.mean()