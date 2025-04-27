# Model Files

Place your trained model files (`.pth`, `.pt`) in this directory.

## Compatible Model Types

The demo application works with the following model architectures:
- UNet++
- FPN
- DeepLabV3+
- PSPNet

All models should be trained using `segmentation_models_pytorch` library for best compatibility.

## Expected Format

Models should be saved as complete PyTorch models, not just state dictionaries.

```python
# Example of proper model saving
torch.save(model, 'models/my_model.pth')
``` 