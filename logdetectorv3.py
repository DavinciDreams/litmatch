import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from PIL import Image
import random
import wandb
import re
from datetime import datetime
import hashlib
import logging
from typing import List, Tuple, Dict, Any, Optional

# --- Config ---
LOG_COUNT = 100000
ANOMALY_RATIO = 0.01
IMAGE_SAVE_PATH = "log_image_nxf.png"
WANDB_PROJECT = "http-log-anomaly-detection"
WANDB_RUN_NAME = "HTTP Log Anomaly Detection - NxF Feature Image"
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'
EPOCHS = 10
BATCH_SIZE = 32
THRESHOLD = 0.08

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def generate_synthetic_logs(n_logs: int = LOG_COUNT, anomaly_ratio: float = ANOMALY_RATIO) -> Tuple[List[str], List[int]]:
    """Generate synthetic HTTP logs with a given anomaly ratio."""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Firefox/89.0",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6) Safari/605.1.15",
        "EvilBot/1.0 (Malicious Bot)"  # Anomaly
    ]
    logs = []
    anomaly_indices = random.sample(range(n_logs), int(n_logs * anomaly_ratio))
    for i in range(n_logs):
        user_agent = user_agents[-1] if i in anomaly_indices else random.choice(user_agents[:-1])
        log = f'127.0.0.1 - - [01/Aug/2025:12:00:00 +0000] "GET /index.html HTTP/1.1" 200 1024 "-" "{user_agent}"'
        logs.append(log)
    return logs, anomaly_indices

def parse_log_features(log: str) -> Optional[Dict[str, Any]]:
    """Parse features from a log line."""
    log_pattern = re.compile(r'(\S+) - - \[(.*?)\] "(\S+) (\S+) (\S+)" (\S+) (\S+) "(.*?)" "(.*?)"')
    match = log_pattern.match(log)
    if match:
        ip_address, timestamp_str, method, requested_path, protocol, status_code, bytes_sent, referer, user_agent = match.groups()
        try:
            timestamp = datetime.strptime(timestamp_str, '%d/%b/%Y:%H:%M:%S %z')
        except ValueError:
            timestamp = None
        try:
            bytes_sent = int(bytes_sent) if bytes_sent != '-' else 0
        except ValueError:
            bytes_sent = 0
        return {
            'ip_address': ip_address, 'timestamp': timestamp, 'method': method,
            'requested_path': requested_path, 'protocol': protocol, 'status_code': status_code,
            'bytes_sent': bytes_sent, 'referer': referer, 'user_agent': user_agent
        }
    return None

def encode_features_to_numerical(parsed_features: Dict[str, Any]) -> List[int]:
    """Encode parsed log features to numerical values."""
    if not parsed_features:
        return [0] * 7
    numerical_features = []
    ip_hash = int(hashlib.sha256(parsed_features['ip_address'].encode('utf-8')).hexdigest(), 16) % (2**16)
    numerical_features.append(ip_hash)
    if parsed_features['timestamp']:
        numerical_features.append(parsed_features['timestamp'].hour)
        numerical_features.append(parsed_features['timestamp'].weekday())
    else:
        numerical_features.extend([0, 0])
    method_mapping = {'GET': 1, 'POST': 2, 'PUT': 3, 'DELETE': 4, 'HEAD': 5, 'OPTIONS': 6}
    numerical_features.append(method_mapping.get(parsed_features['method'], 0))
    try:
        numerical_features.append(int(parsed_features['status_code']))
    except ValueError:
        numerical_features.append(0)
    numerical_features.append(parsed_features['bytes_sent'])
    numerical_features.append(len(parsed_features['user_agent']))
    return numerical_features

def logs_to_nxf_image(logs: List[str], embed_model: SentenceTransformer) -> Tuple[np.ndarray, np.ndarray]:
    """Convert logs to an NxF image (each row = log, columns = features+embedding)."""
    all_features = []
    user_agent_list = []
    for log in logs:
        parsed = parse_log_features(log)
        if parsed:
            all_features.append(encode_features_to_numerical(parsed))
            user_agent_list.append(parsed['user_agent'])
        else:
            all_features.append([0] * 7)
            user_agent_list.append("")
    # Get user agent embeddings
    if user_agent_list:
        embeddings = embed_model.encode(user_agent_list, batch_size=512, show_progress_bar=False)
    else:
        embeddings = np.zeros((len(logs), 384))  # Assuming MiniLM-L6-v2 embedding size
    # Combine numerical features and embeddings
    combined_features = np.hstack([np.array(all_features), np.array(embeddings)])
    # Normalize to 0-255
    min_val = combined_features.min()
    max_val = combined_features.max()
    norm_features = ((combined_features - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return norm_features, embeddings

class Autoencoder(nn.Module):
    """Simple CNN Autoencoder for NxF images."""
    def __init__(self, input_shape: Tuple[int, int]):
        super(Autoencoder, self).__init__()
        channels = 1
        height, width = input_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder(model: nn.Module, image: np.ndarray, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE) -> None:
    """Train autoencoder on NxF image."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0) / 255.0  # [1, 1, N, F]
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(image_tensor)
        loss = criterion(output, image_tensor)
        loss.backward()
        optimizer.step()
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        wandb.log({"train_loss": loss.item()})

def detect_anomalies(model: nn.Module, image: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """Detect anomalies by mapping reconstruction error back to logs."""
    model.eval()
    image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        reconstructed = model(image_tensor)
    error_map = (reconstructed.squeeze().numpy() - image / 255.0) ** 2  # [N, F]
    log_errors = error_map.mean(axis=1)  # Mean error per log
    anomalies = np.where(log_errors > threshold)[0]
    return anomalies

def main():
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)
    logging.info("Generating synthetic logs...")
    logs, true_anomaly_indices = generate_synthetic_logs()
    logging.info("Loading Sentence-BERT model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    logging.info("Converting logs to NxF image...")
    image, embeddings = logs_to_nxf_image(logs, embed_model)
    # Save image for visualization
    img = Image.fromarray(image)
    img.save(IMAGE_SAVE_PATH)
    logging.info("Training autoencoder...")
    model = Autoencoder(input_shape=image.shape)
    train_autoencoder(model, image)
    logging.info("Detecting anomalies...")
    detected_anomalies = detect_anomalies(model, image)
    # Evaluate
    true_positives = len(set(detected_anomalies) & set(true_anomaly_indices))
    precision = true_positives / len(detected_anomalies) if len(detected_anomalies) > 0 else 0
    recall = true_positives / len(true_anomaly_indices) if len(true_anomaly_indices) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    logging.info(f"Detected {len(detected_anomalies)} anomalies.")
    logging.info(f"True anomalies: {len(true_anomaly_indices)}")
    logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    wandb.log({
        "detected_anomalies_count": len(detected_anomalies),
        "true_anomalies_count": len(true_anomaly_indices),
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "note": "Anomaly detection uses NxF image mapping for log-level detection."
    })
    wandb.finish()

if __name__ == "__main__":
    main()