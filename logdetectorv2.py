import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from PIL import Image
import random
import wandb # Import wandb
import re # Import re for parsing
from datetime import datetime # Import datetime for timestamp

# Synthetic HTTP log generator
def generate_synthetic_logs(n_logs=1000000, anomaly_ratio=0.01): # Increased anomaly_ratio
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

# Parse user agent from log (keeping this for now, might be refactored)
def parse_user_agent(log):
    return log.split('"')[-2]

# Define parse_log_features and encode_features_to_numerical functions here or ensure they are defined in previous cells
# Assuming parse_log_features and encode_features_to_numerical are defined and available in the environment

def parse_log_features(log):
    # Regex to parse common log format (adjust if your logs vary)
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

import hashlib

def encode_features_to_numerical(parsed_features):
    if not parsed_features:
        return None
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
    numerical_features.append(len(parsed_features['user_agent'])) # Simple representation for now
    return numerical_features


# Convert logs to image - MODIFIED to use multiple features
def logs_to_image(logs, embed_model, image_size=(1000, 1000)):
    all_features = []
    user_agent_list = []
    for log in logs:
        parsed = parse_log_features(log)
        if parsed:
            all_features.append(encode_features_to_numerical(parsed))
            user_agent_list.append(parsed['user_agent'])
        else:
            # Handle logs that couldn't be parsed - add default features or skip
            all_features.append([0] * 6) # Assuming 6 numerical features for now
            user_agent_list.append("") # Empty user agent

    # Ensure all_features have the same length - pad with zeros if necessary
    max_features = max(len(f) for f in all_features) if all_features else 0
    padded_features = [f + [0] * (max_features - len(f)) for f in all_features]

    # Convert numerical features to numpy array
    numerical_features_np = np.array(padded_features)

    # Get user agent embeddings
    # Handle empty user_agent_list case
    if user_agent_list:
      embeddings = embed_model.encode(user_agent_list, batch_size=512, show_progress_bar=False)
    else:
      embeddings = np.array([]) # Empty array if no logs parsed

    # Combine numerical features and embeddings
    # This is a simple concatenation. More sophisticated methods could be used.
    # We need to ensure the dimensions are compatible for image creation.
    # Let's decide on a strategy:
    # Strategy 1: Create a multi-channel image. Each channel could represent a different feature or a combination.
    # Strategy 2: Flatten features and embeddings for each log entry and arrange them spatially in a grayscale image.
    # Strategy 1 is more complex and requires changing the autoencoder architecture.
    # Strategy 2 is simpler to integrate with the current grayscale autoencoder. Let's try Strategy 2 first.

    # Strategy 2: Flatten features and embeddings and map to grayscale
    # For each log, combine its numerical features and its user agent embedding
    combined_features = []
    for i in range(len(logs)):
        if i < len(numerical_features_np) and i < len(embeddings): # Ensure indices are valid
             combined = np.concatenate((numerical_features_np[i], embeddings[i]))
             combined_features.append(combined)
        else:
             # Handle cases where parsing or embedding failed for a log
             # Append a zero vector of appropriate size
             zero_vector_size = max_features + (embeddings.shape[1] if embeddings.size > 0 else 0)
             combined_features.append(np.zeros(zero_vector_size))


    if not combined_features:
        # Handle case where no logs were processed successfully
        return np.zeros(image_size, dtype=np.uint8), np.array([])

    combined_features_np = np.array(combined_features)

    # Normalize combined features to 0-255 for grayscale image
    # Flatten the combined features array to a 1D array for normalization
    flat_combined_features = combined_features_np.flatten()

    if flat_combined_features.size == 0:
         return np.zeros(image_size, dtype=np.uint8), np.array([])


    pixel_values = (flat_combined_features - flat_combined_features.min()) / (flat_combined_features.max() - flat_combined_features.min()) * 255
    pixel_values = pixel_values.astype(np.uint8)

    # Reshape to image_size (e.g., 1000x1000)
    # This requires the total number of pixel_values to match image_size[0] * image_size[1]
    # If the number of logs * features per log doesn't match image_size, we need padding or a different mapping.
    # Let's pad the pixel_values array to match the required image size.
    required_pixels = image_size[0] * image_size[1]
    if len(pixel_values) < required_pixels:
        padding_size = required_pixels - len(pixel_values)
        pixel_values = np.pad(pixel_values, (0, padding_size), 'constant', constant_values=0)
    elif len(pixel_values) > required_pixels:
         # If we have too many pixel values (more logs than image pixels), we need a different strategy.
         # For this example, let's truncate for simplicity, but this might not be ideal.
         pixel_values = pixel_values[:required_pixels]


    image = pixel_values.reshape(image_size)

    # Return the grayscale image and the original embeddings (if still needed later)
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
def train_autoencoder(model, image, epochs=20, batch_size=1): # Increased epochs
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
        wandb.log({"train_loss": loss.item()}) # Log training loss

# Detect anomalies
def detect_anomalies(model, image, threshold=0.08): # Adjusted threshold
    model.eval()
    image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        reconstructed = model(image_tensor)
    reconstruction_error = torch.mean((reconstructed - image_tensor) ** 2, dim=(1, 2, 3)).numpy()
    # Pixel-wise error
    # Need to reshape reconstructed image back to the original log structure to calculate pixel error per log
    # If the image mapping is a simple 1D array flattened and reshaped, this is complex.
    # Let's stick to overall reconstruction error for simplicity with this image mapping strategy for now.
    # If using pixel_error, need to align pixels back to original logs.
    # For now, let's use the overall image reconstruction error for a simple anomaly score.
    # A log is anomalous if its corresponding pixel(s) in the image have high reconstruction error.
    # With the current flattened mapping, a high error in one pixel might correspond to a combined feature set of a log.

    # Let's reconsider the pixel-wise error and try to map it back to logs.
    # Assuming the pixel_values array was created by flattening combined_features_np row by row (each row is a log's features + embedding)
    # The pixel_error calculated on the reshaped image needs to be re-averaged or analyzed based on the original log structure.

    # Let's try a simpler approach for mapping pixel error to logs:
    # Calculate the absolute error map
    error_map = np.abs(reconstructed.squeeze().numpy() - image / 255.0)

    # We need a way to map parts of the error_map back to individual logs.
    # With the current flattening strategy, this is difficult without knowing the exact mapping logic and dimensions.

    # Alternative: Calculate reconstruction error per log entry.
    # This requires reshaping the reconstructed output to match the original combined_features_np shape before flattening.
    # This is tricky because the autoencoder works on the 2D image.

    # Let's revert to a simpler anomaly detection based on overall image reconstruction error for now,
    # or the pixel-wise error *if* we can reliably map it back to logs.

    # Let's go back to the original pixel_error calculation, but understand its limitation with the current mapping.
    # The np.mean((reconstructed.squeeze().numpy() - image) ** 2, axis=(0, 1)) was incorrect for pixel-wise error on a 2D image.
    # It should be calculating error per pixel and then analyzing those.

    # Correct way to get pixel-wise squared error:
    squared_error_map = (reconstructed.squeeze().numpy() - image / 255.0) ** 2

    # Now, how to relate this 1000x1000 error map back to individual logs?
    # If the image was formed by arranging logs sequentially (e.g., 1000 logs per row for 1000 rows), it's easier.
    # With a single 1000x1000 image from 1,000,000 logs, each pixel represents a fraction of a log or multiple logs are combined into pixels.
    # The current mapping flattens all features/embeddings and reshapes. This loses the per-log structure in the image.

    # Let's change the image mapping strategy to make anomaly detection based on localized error feasible.
    # Strategy: Create a 2D image where each row (or a block of rows) represents a log entry.
    # The columns in that row represent the features (numerical + embedding).
    # If we have N logs and F features per log, we can create an image of size N x F.
    # The autoencoder then needs to handle this NxF size, or we can resize it consistently.

    # Let's modify logs_to_image to create an N x F image.
    # Then, the autoencoder needs to be adapted.

    # Since this requires modifying the autoencoder too, let's postpone this specific pixel-wise error detection
    # until we have the NxF image structure and adapted autoencoder.

    # For now, let's revert detect_anomalies to use a simpler approach based on overall reconstruction error
    # or a threshold on the pixel_error map without precise log mapping.

    # Let's use the mean squared error across the entire image as a simple anomaly score for now.
    mean_squared_error = np.mean(squared_error_map)

    # This approach doesn't give us which *logs* are anomalous, only an overall sense of reconstruction quality.
    # To detect individual log anomalies, we MUST have a mapping from image pixels/regions back to logs.

    # Let's go back to the N x F image strategy. This requires coordinated changes in logs_to_image and Autoencoder.
    # I will update the plan to reflect this.

    # Reverting detect_anomalies for now, as its current pixel_error logic is flawed with the flattening image strategy.
    # We need to implement the NxF image mapping first.

    # Let's return a placeholder for anomalies for now, as the detection logic needs the new image format.
    # This means the evaluation metrics will be based on a faulty detection.
    # I need to inform the user that the anomaly detection part will not work correctly until the image format is updated.

    # For now, I will keep the detect_anomalies function as is, but acknowledge its limitations with the current image mapping.
    # The pixel_error calculation assumes a spatial relationship that doesn't exist with the flattened mapping.
    # The anomalies detected based on this pixel_error threshold might not correspond to individual log anomalies.

    # Let's proceed with the modified logs_to_image for the flattened strategy, but inform the user about the detection limitation.
    # We will need to revisit the plan to address the NxF image mapping and autoencoder adaptation.

    # Keeping the original detect_anomalies for now, but the results will be unreliable for identifying specific log anomalies.
    # The evaluation metrics will reflect this unreliability.

    # --- Keeping original detect_anomalies for now, will fix after image mapping change ---
    # model.eval()
    # image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
    # with torch.no_grad():
    #     reconstructed = model(image_tensor)
    # reconstruction_error = torch.mean((reconstructed - image_tensor) ** 2, dim=(1, 2, 3)).numpy()
    # pixel_error = np.mean((reconstructed.squeeze().numpy() - image / 255.0) ** 2, axis=(0, 1)) # This axis is likely wrong
    # anomalies = np.where(pixel_error > threshold)[0]
    # return anomalies

    # Let's return a simple list of indices based on the overall image reconstruction error for now,
    # as a placeholder for anomaly detection until the image mapping and corresponding detection are fixed.
    # This won't give log-level anomalies but will allow the code to run.

    model.eval()
    image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        reconstructed = model(image_tensor)

    # Calculate the overall mean squared error
    mean_squared_error = torch.mean((reconstructed - image_tensor) ** 2).item()

    # A very simplistic anomaly detection: if the overall error is above a threshold, consider *all* logs potentially anomalous.
    # This is NOT log-level anomaly detection, just a placeholder.
    # We need the NxF image mapping to do proper log-level anomaly detection.

    # Let's return an empty list for detected anomalies for now, to reflect that log-level detection is not yet implemented
    # with the new image mapping strategy. The evaluation metrics will show 0 detected anomalies.
    # This is better than returning misleading anomaly indices from the flawed pixel_error approach.

    return np.array([]) # Placeholder: No log-level anomaly detection yet


# --- Core Anomaly Detection Steps ---

# Initialize wandb run
# Use a different name to indicate the new feature mapping attempt
wandb.init(project="http-log-anomaly-detection", name="HTTP Log Anomaly Detection - Flattened Feature Image")


# Generate synthetic logs
print("Generating synthetic logs...")
logs, true_anomaly_indices = generate_synthetic_logs(anomaly_ratio=0.01) # Use updated anomaly_ratio

# Load Sentence-BERT model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert logs to image
print("Converting logs to image using flattened features...")
image, embeddings = logs_to_image(logs, embed_model)

# Save image (for visualization, optional)
img = Image.fromarray(image, mode='L')
img.save("log_image_flattened_features.png") # Save with a different name

# Initialize and train autoencoder
print("Training autoencoder...")
# Ensure the autoencoder is compatible with the new image size if it's NxF
# With the flattened strategy, the image is still 1000x1000, so the autoencoder should work.
model = Autoencoder()
train_autoencoder(model, image / 255.0, epochs=20)  # Use updated epochs

# Detect anomalies
print("Detecting anomalies (using placeholder logic)...")
# Note: The detect_anomalies function currently uses a placeholder as log-level detection
# needs the NxF image mapping strategy.
detected_anomalies = detect_anomalies(model, image / 255.0, threshold=0.08) # Use updated threshold

# Evaluate
# The evaluation metrics will be misleading as detected_anomalies is a placeholder.
true_positives = len(set(detected_anomalies) & set(true_anomaly_indices))
precision = true_positives / len(detected_anomalies) if detected_anomalies.size > 0 else 0
recall = true_positives / len(true_anomaly_indices) if true_anomaly_indices else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f"Detected {len(detected_anomalies)} anomalies.")
print(f"True anomalies: {len(true_anomaly_indices)}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Log evaluation metrics
# Log a warning or note in wandb that detection is placeholder
wandb.log({
    "detected_anomalies_count": len(detected_anomalies),
    "true_anomalies_count": len(true_anomaly_indices),
    "true_positives": true_positives,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "note": "Anomaly detection is using a placeholder due to image mapping strategy. Log-level detection needs NxF image."
})

# Finish wandb run
wandb.finish()