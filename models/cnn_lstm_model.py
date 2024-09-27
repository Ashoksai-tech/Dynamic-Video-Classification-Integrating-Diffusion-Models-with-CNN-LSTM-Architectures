import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Calculate the output size of CNN
        self.cnn_output_size = hidden_channels * 2 * 3 * 3  # Assuming input is 14x14, after two MaxPool2d it becomes 3x3
        
        self.lstm = nn.LSTM(self.cnn_output_size, hidden_channels, batch_first=True)
        self.fc = nn.Linear(hidden_channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, c_n) = self.lstm(r_in)
        r_out2 = self.fc(r_out[:, -1, :])
        return self.sigmoid(r_out2)