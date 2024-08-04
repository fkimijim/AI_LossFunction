class TverskyLoss(nn.Module):
    def __init__(self, alpha=1, beta=0.6, smooth=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        assert len(inputs.shape) == 4, "logitsの形状は(b, c, h, w)です"
        inputs = torch.sigmoid(inputs)

        batch_loss = 0.0
        for i in range(inputs.size(0)):
            input_flat = inputs[i].view(-1)
            target_flat = targets[i].view(-1)

            TP = (input_flat * target_flat).sum()
            FP = ((1 - target_flat) * input_flat).sum()
            FN = (target_flat * (1 - input_flat)).sum()

            tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
            tversky = 1 - tversky

            batch_loss += tversky

        batch_loss /= inputs.size(0)
        return batch_loss