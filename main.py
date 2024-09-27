import torch
import yaml
import warnings
from models.diffusion_model import DiffusionModel, compute_reconstruction_error
from models.cnn_lstm_model import CNNLSTM
from utils.video_loader import load_video
from utils.preprocessing import preprocess_video, augment_video
from utils.evaluation import evaluate_binary_classification, print_evaluation_results

warnings.filterwarnings("ignore", category=UserWarning)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config('config.yaml')
    
    # Load and preprocess video
    video = load_video(config['video_path'], config['num_frames'])
    processed_video = preprocess_video(video, (config['frame_height'], config['frame_width']))
    video_tensor = torch.FloatTensor(processed_video).unsqueeze(0)  # Add batch dimension
    
    # Augment video (for demonstration purposes)
    augmented_video = augment_video(video)
    processed_augmented_video = preprocess_video(augmented_video, (config['frame_height'], config['frame_width']))
    augmented_video_tensor = torch.FloatTensor(processed_augmented_video).unsqueeze(0)
    
    # Initialize models
    diffusion_model = DiffusionModel(config['input_channels'], config['hidden_channels'])
    
    # Process videos through diffusion model
    with torch.no_grad():
        encoded_features, reconstructed = diffusion_model(video_tensor)
        reconstruction_error = compute_reconstruction_error(video_tensor, reconstructed)
        
        encoded_features_aug, reconstructed_aug = diffusion_model(augmented_video_tensor)
        reconstruction_error_aug = compute_reconstruction_error(augmented_video_tensor, reconstructed_aug)
    
    # Print shapes for debugging
    print(f"Encoded features shape: {encoded_features.shape}")
    print(f"Reconstruction error shape: {reconstruction_error.shape}")
    
    # Reshape reconstruction error to match encoded features
    b, t, c, h, w = encoded_features.shape
    reconstruction_error = reconstruction_error.view(b, t, 1, 1, 1).expand(b, t, 1, h, w)
    reconstruction_error_aug = reconstruction_error_aug.view(b, t, 1, 1, 1).expand(b, t, 1, h, w)
    
    # Combine encoded features and reconstruction error
    combined_features = torch.cat([encoded_features, reconstruction_error], dim=2)
    combined_features_aug = torch.cat([encoded_features_aug, reconstruction_error_aug], dim=2)
    
    # Initialize CNNLSTM model with correct input size
    cnn_lstm_model = CNNLSTM(combined_features.size(2), config['hidden_channels'])
    
    # Process through CNN+LSTM model
    with torch.no_grad():
        output = cnn_lstm_model(combined_features)
        output_aug = cnn_lstm_model(combined_features_aug)
    
    # Convert to binary prediction
    binary_prediction = 1 if output.item() > 0.5 else 0
    binary_prediction_aug = 1 if output_aug.item() > 0.5 else 0
    
    print(f"Original video - Model output: {output.item():.4f}, Binary prediction: {binary_prediction}")
    print(f"Augmented video - Model output: {output_aug.item():.4f}, Binary prediction: {binary_prediction_aug}")
    
    # Simulated evaluation (for demonstration purposes)
    y_true = [1, 0]  # Assuming original video should be classified as 1, augmented as 0
    y_pred = [binary_prediction, binary_prediction_aug]
    y_pred_proba = [output.item(), output_aug.item()]
    
    metrics = evaluate_binary_classification(y_true, y_pred, y_pred_proba)
    print_evaluation_results(metrics)

if __name__ == "__main__":
    main()