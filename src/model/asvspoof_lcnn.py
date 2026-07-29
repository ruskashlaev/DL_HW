import torch
from torch import nn


class MFM(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return torch.max(a, b)


class LCNN(nn.Module):
    def __init__(self, n_class=2, n_mels=80, time_frames=751, dropout=0.75):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0),
            MFM(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            MFM(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        flatten_size = self._get_flatten_size(n_mels, time_frames)

        self.fc_layers = nn.Sequential(
            nn.Linear(flatten_size, 160),
            MFM(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(80),
            nn.Linear(80, n_class),
        )

    def _get_flatten_size(self, n_mels, time_frames):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, time_frames)
            out = self.conv_layers(dummy)
            return out.flatten(1).shape[1]

    def forward(self, data_object, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        x = data_object.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.flatten(1)
        logits = self.fc_layers(x)
        return {"logits": logits}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
