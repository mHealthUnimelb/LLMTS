from torch import nn

class base_Model_simclr(nn.Module):
    def __init__(self, configs):
        super(base_Model_simclr, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # self.conv_block4 = nn.Sequential(
        #     nn.Conv1d(configs.final_out_channels, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
        #     nn.BatchNorm1d(configs.final_out_channels),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        #     nn.Dropout(configs.dropout)
        # )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)
        # self.logits = nn.Linear(20352, configs.num_classes) ##for discussing the ecg

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model_output_dim * configs.final_out_channels, 2048),
            # nn.Linear(20352, 2048),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2048, eps=0.001),
            nn.Linear(2048, 64) ##encoding size = 64

        )


        self.pre = nn.Linear(64, configs.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # x = self.conv_block4(x)

        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)

        encoding = self.fc(x_flat)  ## feature...


        return logits, encoding
