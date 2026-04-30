"""
NEURAL STYLE TRANSFER
Apply artistic styles to photographs using PyTorch + VGG19.
Copy-paste this entire file into VS Code and run it.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import os


# ==================== CONFIGURATION ====================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {DEVICE}")

# VGG19 layers for content and style extraction
CONTENT_LAYERS = ['conv_4']          # Layer to preserve content
STYLE_LAYERS = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']  # Layers for style

# Weights
STYLE_WEIGHT = 1e6        # How strong the style effect is (higher = more artistic)
CONTENT_WEIGHT = 1        # How much original content is preserved

# Training settings
NUM_STEPS = 300           # More steps = better quality (300 is good balance)
IMAGE_SIZE = 512          # Output image size (reduce if you run out of memory)


# ==================== IMAGE UTILITIES ====================

def load_image(image_path, max_size=IMAGE_SIZE):
    """Load and preprocess an image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert('RGB')

    # Resize maintaining aspect ratio
    size = max(image.size)
    if size > max_size:
        size = max_size

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Add batch dimension and move to device
    image = transform(image).unsqueeze(0)
    return image.to(DEVICE, torch.float)


def tensor_to_image(tensor):
    """Convert tensor back to PIL Image."""
    image = tensor.clone().detach().cpu().numpy()
    image = image.squeeze(0)
    image = image.transpose(1, 2, 0)

    # Denormalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = std * image + mean
    image = (image * 255).clip(0, 255).astype('uint8')

    return Image.fromarray(image)


# ==================== LOSS FUNCTIONS ====================

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, x):
        G = self.gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

    @staticmethod
    def gram_matrix(x):
        batch_size, n_features, height, width = x.size()
        features = x.view(batch_size * n_features, height * width)
        G = torch.mm(features, features.t())
        return G.div(batch_size * n_features * height * width)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, x):
        return (x - self.mean) / self.std


# ==================== MODEL SETUP ====================

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=CONTENT_LAYERS,
                               style_layers=STYLE_LAYERS):
    """Build the style transfer model with loss layers inserted."""
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(DEVICE)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Trim layers after the last loss layer
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]

    return model, style_losses, content_losses


def run_style_transfer(content_img, style_img, input_img, num_steps=NUM_STEPS,
                       style_weight=STYLE_WEIGHT, content_weight=CONTENT_WEIGHT):
    """Run the neural style transfer optimization."""
    print("⏳ Loading pre-trained VGG19...")
    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(DEVICE).eval()

    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)

    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img
    )

    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_img])

    print(f"🎨 Starting style transfer ({num_steps} steps)...")
    print("   (This may take 1-5 minutes depending on your hardware)\n")

    run = [0]
    while run[0] <= num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score = sum(sl.loss for sl in style_losses) * style_weight
            content_score = sum(cl.loss for cl in content_losses) * content_weight
            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"   Step [{run[0]}/{num_steps}]  Style Loss: {style_score.item():.2f}  |  Content Loss: {content_score.item():.2f}")

            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    print("\n✅ Style transfer complete!")
    return input_img


# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("   🎨 NEURAL STYLE TRANSFER")
    print("=" * 60)

    # Get file paths from user
    print("\n📁 Enter the full path to your images:")
    content_path = input("Content image (photo): ").strip().strip('"')
    style_path = input("Style image (artwork): ").strip().strip('"')

    # Load images
    print("\n⏳ Loading images...")
    content_img = load_image(content_path)
    style_img = load_image(style_path)

    # Ensure same size
    if content_img.shape != style_img.shape:
        print("⚠️  Resizing style image to match content image...")
        style_img = torch.nn.functional.interpolate(
            style_img, size=content_img.shape[2:], mode='bilinear', align_corners=False
        )

    # Start with content image (better results than random noise)
    input_img = content_img.clone()

    # Run transfer
    output = run_style_transfer(content_img, style_img, input_img)

    # Save result
    output_dir = os.path.dirname(content_path) or "."
    output_path = os.path.join(output_dir, "stylized_output.jpg")
    output_image = tensor_to_image(output)
    output_image.save(output_path, quality=95)

    print(f"\n💾 Saved stylized image to: {output_path}")
    print("🎉 Done! Open the image to see the result.")


if __name__ == "__main__":
    main()
