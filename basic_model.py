import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
image_size = 128
message_length = 30
batch_size = 8

# Dataset class to load images from a folder
class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.JPEG'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Transformations for the images
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# Load dataset
dataset = ImageDataset(folder_path="C:/S25/fyp scheme/VitTransformersNewWScheme/data/train/train_class", transform=transform)
# print(f"Total images loaded: {len(dataset)}")

# # Print the first few image file names to verify
# print("Sample images:", dataset.image_files[:5])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Traditional LSB-based Encoder
class TraditionalEncoder(nn.Module):
    def __init__(self):
        super(TraditionalEncoder, self).__init__()
        self.message_length = message_length

    def forward(self, image, message):
        """
        Embeds the message into the LSB of the image pixels.
        Args:
            image: Input image tensor of shape [batch_size, 3, H, W].
            message: Binary message tensor of shape [batch_size, message_length].
        Returns:
            watermarked_image: Image with the message embedded in the LSB.
        """
        batch_size, _, H, W = image.shape

        # Reshape the message to match the image dimensions
        message = message.view(batch_size, 1, 1, -1)  # [batch_size, 1, 1, message_length]
        message = message.repeat(1, 3, H, W // self.message_length)  # Repeat message across channels and spatial dimensions

        # Convert the message to the same dtype as the image
        message = message.to(image.dtype)

        # Embed the message in the LSB of the image
        watermarked_image = image.clone()
        watermarked_image[:, :, :, :self.message_length] = (watermarked_image[:, :, :, :self.message_length] & 0xFE) | message

        return watermarked_image

# Traditional LSB-based Decoder
class TraditionalDecoder(nn.Module):
    def __init__(self):
        super(TraditionalDecoder, self).__init__()
        self.message_length = message_length

    def forward(self, image):
        """
        Extracts the message from the LSB of the image pixels.
        Args:
            image: Watermarked image tensor of shape [batch_size, 3, H, W].
        Returns:
            message: Extracted binary message tensor of shape [batch_size, message_length].
        """
        batch_size, _, H, W = image.shape

        # Extract the LSBs from the image
        message = image[:, :, :, :self.message_length] & 0x01  # Extract LSBs
        message = message.view(batch_size, -1)  # Reshape to [batch_size, message_length]

        return message

# Noise layers
class Identity(nn.Module):
    """
    Identity noise layer (no noise is applied).
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, encoded_and_cover):
        return encoded_and_cover[0]  # Return only the encoded image

class JpegCompression(nn.Module):
    """
    Simulates JPEG compression noise.
    """
    def __init__(self, device):
        super(JpegCompression, self).__init__()
        self.device = device

    def forward(self, encoded_and_cover):
        # Simulate JPEG compression (placeholder implementation)
        encoded_image = encoded_and_cover[0]
        return encoded_image  # No actual compression in this placeholder

class Quantization(nn.Module):
    """
    Simulates quantization noise.
    """
    def __init__(self, device):
        super(Quantization, self).__init__()
        self.device = device

    def forward(self, encoded_and_cover):
        # Simulate quantization (placeholder implementation)
        encoded_image = encoded_and_cover[0]
        return encoded_image  # No actual quantization in this placeholder

# Noiser module
class Noiser(nn.Module):
    """
    Applies a random noise layer to the watermarked image.
    """
    def __init__(self, noise_layers: list, device):
        super(Noiser, self).__init__()
        self.noise_layers = [Identity()]
        for layer in noise_layers:
            if type(layer) is str:
                if layer == 'JpegPlaceholder':
                    self.noise_layers.append(JpegCompression(device))
                elif layer == 'QuantizationPlaceholder':
                    self.noise_layers.append(Quantization(device))
                else:
                    raise ValueError(f'Wrong layer placeholder string in Noiser._init_().'
                                     f' Expected "JpegPlaceholder" or "QuantizationPlaceholder" but got {layer} instead')
            else:
                self.noise_layers.append(layer)

    def forward(self, encoded_and_cover):
        random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        return random_noise_layer(encoded_and_cover)

# Loss functions
def encoder_mse(cover_image, watermarked_image):
    """
    Computes the mean squared error between the cover image and the watermarked image.
    """
    return torch.mean((cover_image - watermarked_image) ** 2)

def decoder_mse(original_message, decoded_message):
    """
    Computes the mean squared error between the original message and the decoded message.
    """
    return torch.mean((original_message - decoded_message) ** 2)

def bitwise_error(original_message, decoded_message):
    """
    Computes the bitwise error (Hamming distance) between the original message and the decoded message.
    """
    return torch.mean(torch.abs(original_message - decoded_message))

# Initialize models
traditional_encoder = TraditionalEncoder().to(device)
traditional_decoder = TraditionalDecoder().to(device)

# Initialize Noiser
noise_layers = ['JpegPlaceholder', 'QuantizationPlaceholder']  # Add noise layers here
noiser = Noiser(noise_layers, device).to(device)

# Evaluation loop
for i, images in enumerate(dataloader):
    images = images.to(device)
    batch_size = images.size(0)

    # Generate random binary messages
    messages = torch.randint(0, 2, (batch_size, message_length), dtype=torch.float32).to(device)

    # Traditional LSB-based Watermarking
    watermarked_images = traditional_encoder(images, messages)

    # Apply noise to the watermarked images
    noised_images = noiser([watermarked_images, images])

    # Decode the message from the noised images
    decoded_messages = traditional_decoder(noised_images)

    # Compute metrics
    encoder_mse_value = encoder_mse(images, watermarked_images)
    decoder_mse_value = decoder_mse(messages, decoded_messages)
    bitwise_error_value = bitwise_error(messages, decoded_messages)

    # Print results
    print(f"Batch [{i+1}/{len(dataloader)}]")
    print(f"Encoder MSE: {encoder_mse_value.item():.4f}, Decoder MSE: {decoder_mse_value.item():.4f}, Bitwise Error: {bitwise_error_value.item():.4f}")