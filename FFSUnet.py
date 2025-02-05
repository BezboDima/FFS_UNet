from typing import List, Tuple
import torch
import torch.nn as nn
from ResT.ResT.models.rest_v2 import Block
from pytorchcv.models import shufflenetv2
from transformers import ViTModel


class SimplifiedFFS_UNet(nn.Module):
    def __init__(self) -> None:
        """Initializes the SimplifiedFFS_UNet model.

        The network uses three parallel downsampling paths, a bottleneck, and an upsampling module.
        """
        super().__init__()
        self.fn_minus2 = DownsamplingSimple()
        self.fn = DownsamplingSimple()
        self.fn_plus2 = DownsamplingSimple()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(2112, 2112, kernel_size=3, padding=1),  # Maintain dimension
            nn.BatchNorm2d(2112),
            nn.ReLU(),
            nn.Conv2d(2112, 704, kernel_size=1)  # Final projection
        )
        self.upsample = Upsampling([704, 352, 176, 24, 24])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor with shape (B, 3, H, W), where the second dimension
                contains three images (channels split along dim=1).

        Returns:
            torch.Tensor: Output segmentation mask.
        """
        images: List[torch.Tensor] = torch.unbind(x, dim=1)

        fn_minus2, _ = self.fn_minus2(images[0])
        fn, residuals = self.fn(images[1])
        fn_plus2, _ = self.fn_plus2(images[2])

        concatenated_features: torch.Tensor = torch.cat([fn_minus2, fn, fn_plus2], dim=1)
        x = self.bottleneck(concatenated_features)
        x = self.upsample(x, residuals)
        return x


class FFS_UNet(nn.Module):
    def __init__(self) -> None:
        """Initializes the FFS_UNet model.

        Uses three downsampling paths, a temporal transformer module, and an upsampling module.
        """
        super(FFS_UNet, self).__init__()
        self.fn_minus2 = Downsampling()
        self.fn = Downsampling()
        self.fn_plus2 = Downsampling()
        self.ttm = TemporalTransformerModulePretrained(dim=704, num_heads=8, num_blocks=2, sr_ratio=1)
        self.upsample = Upsampling([704, 352, 176, 24, 24])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor with shape (B, 3, H, W) where the second dimension
                represents three temporal images.

        Returns:
            torch.Tensor: Output segmentation mask after applying sigmoid activation.
        """
        images: List[torch.Tensor] = torch.unbind(x, dim=1)

        fn_minus2, _ = self.fn_minus2(images[0])
        fn, residuals = self.fn(images[1])
        fn_plus2, _ = self.fn_plus2(images[2])

        x = self.ttm(fn_minus2, fn, fn_plus2)
        x = self.upsample(x, residuals)
        x = torch.sigmoid(x)
        return x


class Downsampling(nn.Module):
    def __init__(self) -> None:
        """Initializes the Downsampling module.

        The module applies a series of convolutional and pooling layers followed by three stages using
        ShuffleNet blocks.
        """
        super(Downsampling, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = ShuffleNet_Stage(24, 176, repeats=3)
        self.stage3 = ShuffleNet_Stage(176, 352, repeats=7)
        self.stage4 = ShuffleNet_Stage(352, 704, repeats=3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Performs a forward pass through the downsampling module.

        Args:
            x (torch.Tensor): Input tensor with shape (B, 3, H, W).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - Final features from stage 4.
                - A list of intermediate features (for skip connections) from stage3, stage2, maxpool, and conv1.
        """
        x_conv: torch.Tensor = self.conv1(x)
        x_max: torch.Tensor = self.maxpool(x_conv)
        x_stage2: torch.Tensor = self.stage2(x_max)
        x_stage3: torch.Tensor = self.stage3(x_stage2)
        x_out: torch.Tensor = self.stage4(x_stage3)
        return x_out, [x_stage3, x_stage2, x_max, x_conv]


class DownsamplingSimple(nn.Module):
    def __init__(self) -> None:
        """Initializes the DownsamplingSimple module.

        This module provides a simplified version of the downsampling path with sequential convolutional blocks.
        """
        super(DownsamplingSimple, self).__init__()
        # Initial downsampling
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Simplified downsampling blocks
        self.stage2 = self._make_down_block(24, 176)
        self.stage3 = self._make_down_block(176, 352)
        self.stage4 = self._make_down_block(352, 704)

    def _make_down_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Creates a downsampling block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Sequential: A sequential block with convolution, batch normalization, and ReLU.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Performs a forward pass through the simplified downsampling module.

        Args:
            x (torch.Tensor): Input tensor with shape (B, 3, H, W).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - Final features from stage 4.
                - A list of intermediate features (for skip connections) from stage3, stage2, maxpool, and conv1.
        """
        x_conv: torch.Tensor = self.conv1(x)
        x_max: torch.Tensor = self.maxpool(x_conv)
        x_stage2: torch.Tensor = self.stage2(x_max)
        x_stage3: torch.Tensor = self.stage3(x_stage2)
        x_stage4: torch.Tensor = self.stage4(x_stage3)
        return x_stage4, [x_stage3, x_stage2, x_max, x_conv]


class Upsampling_Stage(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        """Initializes an upsampling stage.

        Args:
            input_channels (int): Number of input channels after concatenation.
            output_channels (int): Number of output channels after convolution.
        """
        super(Upsampling_Stage, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Performs an upsampling operation and concatenates skip connections.

        Args:
            x (torch.Tensor): The feature map to upsample.
            y (torch.Tensor): The skip connection feature map.

        Returns:
            torch.Tensor: The resulting feature map after upsampling, concatenation, convolution, and activation.
        """
        x = self.upsample(x)
        concatenated: torch.Tensor = torch.cat([x, y], dim=1)
        x = self.conv(concatenated)
        x = self.relu(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, channels_list: List[int]) -> None:
        """Initializes the Upsampling module.

        Args:
            channels_list (List[int]): List of channel sizes at different stages.
        """
        super(Upsampling, self).__init__()
        self.upsampling_stages = nn.ModuleList()

        for index in range(len(channels_list) - 1):
            self.upsampling_stages.append(
                Upsampling_Stage(
                    input_channels=channels_list[index] + channels_list[index + 1],
                    output_channels=channels_list[index + 1]
                )
            )
        self.last_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv11 = nn.Conv2d(
            in_channels=24,
            out_channels=1,
            kernel_size=1,
            bias=True  # Enable bias
        )

    def forward(self, x: torch.Tensor, y: List[torch.Tensor]) -> torch.Tensor:
        """Performs a forward pass through the upsampling module.

        Args:
            x (torch.Tensor): The input feature map from the bottleneck.
            y (List[torch.Tensor]): A list of skip connection feature maps.

        Returns:
            torch.Tensor: The final upsampled output (segmentation mask).
        """
        for stage, res in zip(self.upsampling_stages, y):
            x = stage(x, res)
        x = self.last_upsample(x)
        x = self.conv11(x)
        return x


class ShuffleNet_Stage(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, repeats: int) -> None:
        """Initializes a stage of ShuffleNet units.

        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            repeats (int): Number of repeated ShuffleNet units.
        """
        super(ShuffleNet_Stage, self).__init__()

        # First block with downsampling (stride=2)
        self.shuffle_netV2_b = shufflenetv2.ShuffleUnit(
            input_channels, output_channels, downsample=True, use_se=False, use_residual=False
        )

        # Create a list to hold the repeated blocks
        self.repeated_blocks = nn.ModuleList()

        # Add the repeated blocks
        for _ in range(repeats):
            self.repeated_blocks.append(
                shufflenetv2.ShuffleUnit(
                    output_channels, output_channels, downsample=False, use_se=False, use_residual=False
                )
            )

        # Unfreeze all parameters in the ShuffleNet_Stage module
        for param in self.shuffle_netV2_b.parameters():
            param.requires_grad = True

        for block in self.repeated_blocks:
            for param in block.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the ShuffleNet stage.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the ShuffleNet blocks.
        """
        x = self.shuffle_netV2_b(x)
        for block in self.repeated_blocks:
            x = block(x)
        return x


class TemporalTransformerModulePretrained(nn.Module):
    def __init__(self, input_channels: int = 704, output_channels: int = 704) -> None:
        """Initializes the TemporalTransformerModulePretrained.

        Loads a pretrained ViT model and defines input/output projection layers.

        Args:
            input_channels (int, optional): Number of input channels. Defaults to 704.
            output_channels (int, optional): Number of output channels after projection. Defaults to 704.
        """
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.input_proj = nn.Linear(input_channels, self.vit.config.hidden_size)
        self.output_proj = nn.Conv2d(self.vit.config.hidden_size, output_channels, kernel_size=1)

    def forward(self, F_n_minus_2: torch.Tensor, F_n: torch.Tensor, F_n_plus_2: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass using a pretrained ViT.

        Args:
            F_n_minus_2 (torch.Tensor): Feature map from time step n-2.
            F_n (torch.Tensor): Feature map from time step n.
            F_n_plus_2 (torch.Tensor): Feature map from time step n+2.

        Returns:
            torch.Tensor: The output feature map after transformer processing.
        """
        weights: torch.Tensor = torch.tensor([0.33, 0.33, 0.33], device=F_n.device)
        concatenated_features: torch.Tensor = (
            weights[0] * F_n_minus_2 + weights[1] * F_n + weights[2] * F_n_plus_2
        )
        B, C, H, W = concatenated_features.shape 

        # Reshape and project to ViT's hidden size
        features: torch.Tensor = concatenated_features.flatten(2).transpose(1, 2)
        features = self.input_proj(features)  

        # Add positional embeddings (ignoring [CLS] token)
        positional_embeddings: torch.Tensor = self.vit.embeddings.position_embeddings[:, 1:, :]  
        features = features + positional_embeddings.to(features.device)

        # Pass through ViT encoder
        encoder_outputs = self.vit.encoder(features)
        encoder_output: torch.Tensor = encoder_outputs.last_hidden_state 

        # Reshape and project to output channels
        encoder_output = encoder_output.transpose(1, 2).reshape(B, -1, H, W)
        transformer_output: torch.Tensor = self.output_proj(encoder_output) 

        return transformer_output # (B, output_channels, H, W)


class TemporalTransformerModule(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_blocks: int, sr_ratio: int, input_channels: int = 704, output_channels: int = 704) -> None:
        """Initializes the TemporalTransformerModule.

        Applies a series of transformer blocks to concatenated features and then projects them.

        Args:
            dim (int): Dimensionality of the input features.
            num_heads (int): Number of attention heads.
            num_blocks (int): Number of transformer blocks.
            sr_ratio (int): Spatial reduction ratio.
            input_channels (int, optional): Number of input channels. Defaults to 704.
            output_channels (int, optional): Number of output channels after projection. Defaults to 704.
        """
        super().__init__()
        self.temporal_transformer_blocks = nn.ModuleList([
            Block(dim=dim, num_heads=num_heads, sr_ratio=sr_ratio)
            for _ in range(num_blocks)
        ])

        self.projection = nn.Conv2d(
            in_channels=input_channels,  # Input channels from concatenated features
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self, F_n_minus_2: torch.Tensor, F_n: torch.Tensor, F_n_plus_2: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the temporal transformer.

        Args:
            F_n_minus_2 (torch.Tensor): Feature map from time step n-2.
            F_n (torch.Tensor): Feature map from time step n.
            F_n_plus_2 (torch.Tensor): Feature map from time step n+2.

        Returns:
            torch.Tensor: The output feature map after transformer processing and projection.
        """
        concatenated_features: torch.Tensor = torch.cat([F_n_minus_2, F_n, F_n_plus_2], dim=1)
        B, C, H, W = concatenated_features.shape
        concatenated_features = concatenated_features.flatten(2).transpose(1, 2)  # Shape: (B, 196, C)

        for blk in self.temporal_transformer_blocks:
            concatenated_features = blk(concatenated_features, H, W)

        transformer_output: torch.Tensor = concatenated_features.transpose(1, 2).reshape(B, -1, H, W)
        transformer_output = self.projection(transformer_output)
        return transformer_output



input_tensor = torch.randn(1 ,3, 3, 448, 448)  # Batch size=1, image=3, Channels=3, Height=448, Width=448
model = FFS_UNet()
output_tensor = model(input_tensor)
print(output_tensor.shape)  # Expected output: torch.Size([1, input_channels, 112, 112])