import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from PIL import Image
import random
import asyncio
import platform

# Synthetic HTTP log generator
def generate_synthetic_logs(n_logs=1000000, anomaly_ratio=0.00001):
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Firefox/89.0",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6) Safari/605.1.15",
        "EvilBot/1.0 (Malicious Bot)"  # Anomaly
    ]
    logs = []
    anomaly_indices = random.sample(range(n_logs), int(n_logs * anomaly_ratio))
    for i in range(n_logs):
        if i in anomaly_indices:
            user_agent = user_agents[-1]  # Anomaly
        else:
            user_agent = random.choice(user_agents[:-1])  # Normal
        log = f'127.0.0.1 - - [01/Aug/2025:12:00:00 +0000] "GET /index.html HTTP/1.1" 200 1024 "-" "{user_agent}"'
        logs.append(log)
    return logs, anomaly_indices

# Parse user agent from log
def parse_user_agent(log):
    return log.split('"')[-2]

# Convert logs to image
def logs_to_image(logs, embed_model, image_size=(1000, 1000)):
    user_agents = [parse_user_agent(log) for log in logs]
    embeddings = embed_model.encode(user_agents, batch_size=512, show_progress_bar=False)
    # Reduce embeddings to a single value per log (mean of embedding)
    pixel_values = np.mean(embeddings, axis=1)
    # Normalize to 0-255 for grayscale image
    pixel_values = (pixel_values - pixel_values.min()) / (pixel_values.max() - pixel_values.min()) * 255
    pixel_values = pixel_values.astype(np.uint8)
    # Reshape to 1000x1000
    image = pixel_values.reshape(image_size)
    return image, embeddings

# CNN Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Train autoencoder
def train_autoencoder(model, image, epochs=10, batch_size=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)  # [1, 1, 1000, 1000]
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(image_tensor)
        loss = criterion(output, image_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Detect anomalies
def detect_anomalies(model, image, threshold=0.1):
    model.eval()
    image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        reconstructed = model(image_tensor)
    reconstruction_error = torch.mean((reconstructed - image_tensor) ** 2, dim=(1, 2, 3)).numpy()
    # Pixel-wise error
    pixel_error = np.mean((reconstructed.squeeze().numpy() - image) ** 2, axis=(0, 1))
    anomalies = np.where(pixel_error > threshold)[0]
    return anomalies

async def main():
    # Generate synthetic logs
    print("Generating synthetic logs...")
    logs, true_anomaly_indices = generate_synthetic_logs()
    
    # Load Sentence-BERT model
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Convert logs to image
    print("Converting logs to image...")
    image, embeddings = logs_to_image(logs, embed_model)
    
    # Save image (for visualization, optional)
    img = Image.fromarray(image, mode='L')
    img.save("log_image.png")
    
    # Initialize and train autoencoder
    print("Training autoencoder...")
    model = Autoencoder()
    train_autoencoder(model, image / 255.0)  # Normalize image for training
    
    # Detect anomalies
    print("Detecting anomalies...")
    detected_anomalies = detect_anomalies(model, image / 255.0)
    
    # Evaluate
    true_positives = len(set(detected_anomalies) & set(true_anomaly_indices))
    precision = true_positives / len(detected_anomalies) if detected_anomalies.size > 0 else 0
    recall = true_positives / len(true_anomaly_indices) if true_anomaly_indices else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Detected {len(detected_anomalies)} anomalies.")
    print(f"True anomalies: {len(true_anomaly_indices)}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())