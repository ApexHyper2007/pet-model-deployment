# Oxford-IIIT Pet Breed Classification

A deep learning image classification project that identifies the breed of a cat or dog from an input image.

The project uses the Oxford-IIIT Pet dataset and a pretrained ResNet-18 model with transfer learning and fine-tuning. The trained model can be used from the command line or through a FastAPI service.

## Project Overview

The model classifies pet images into 37 different cat and dog breeds.

### Dataset

- Dataset: Oxford-IIIT Pet Dataset
- Number of classes: 37
- Images used: 7,349
- Image size: 224 × 224
- Training images: 5,141
- Validation images: 1,096
- Test images: 1,112

## Machine Learning Pipeline

The project follows this pipeline:

1. Dataset verification
2. Image preprocessing
3. Image resizing to 224 × 224
4. Train/validation/test splitting
5. Data augmentation
6. Image normalization
7. Transfer learning using ResNet18
8. Fine-tuning
9. Model evaluation
10. Breed prediction on new images

## Preprocessing

All images were resized to:

```text
224 × 224 pixels
```

Images are converted to tensors and normalized using ImageNet mean and standard deviation:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

## Data Augmentation

During training, the following augmentations are applied:

- Random horizontal flipping
- Random rotation up to 15 degrees
- Random brightness adjustment
- Random contrast adjustment

Augmentation is performed during training rather than creating additional image files.

## Model

The project uses **ResNet18** with pretrained ImageNet weights.

Instead of training the entire network from scratch:

- Most pretrained layers are frozen.
- The final ResNet block (`layer4`) is fine-tuned.
- The final fully connected layer is replaced with a classifier for 37 classes.

This approach uses transfer learning to adapt a pretrained image-recognition model to the pet-breed classification task.

## Training

The model was trained using:

- Framework: PyTorch
- Optimizer: AdamW
- Learning rates: 0.00001 for `layer4`, 0.0001 for the classifier
- Batch size: 32
- Loss function: CrossEntropyLoss
- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU

The current best fine-tuned checkpoint is saved as:

```text
resnet18_finetuned_best.pth
```

The best validation accuracy achieved during fine-tuning was:

```text
90.33%
```

## Test Results

The fine-tuned checkpoint was evaluated once on the held-out test set.

```text
Test images:       1112
Correct:           996
Test accuracy:     89.57%
```

Overall evaluation:

| Metric | Score |
|---|---:|
| Accuracy | 0.90 |
| Macro F1 | 0.89 |
| Weighted F1 | 0.90 |

## Project Structure

```text
oxford-iiit-pet/
│
├── annotations/
├── images/
├── processed_images/
├── old_scripts/
│
├── dataset.py
├── model.py
├── train.py
├── test.py
├── evaluate.py
├── predict.py
├── api.py
├── Dockerfile
├── .dockerignore
│
├── train.txt
├── validation.txt
├── test.txt
│
└── resnet18_finetuned_best.pth
```

### Main Files

| File | Purpose |
|---|---|
| `dataset.py` | Loads images, applies augmentation and normalization |
| `model.py` | Defines the ResNet18 model |
| `train.py` | Trains and validates the model |
| `test.py` | Evaluates accuracy on the test set |
| `evaluate.py` | Generates detailed classification metrics |
| `predict.py` | Predicts the breed of a new image |
| `api.py` | Provides the FastAPI prediction endpoint |

The `old_scripts/` folder contains scripts used during earlier stages of dataset checking and experimentation.

## Running the Project

The commands below can be run from the project directory using the existing Python environment.

### Train

```powershell
python train.py
```

### Test

```powershell
python test.py
```

### Detailed Evaluation

```powershell
python evaluate.py
```

### Predict a New Image

```powershell
python predict.py image.jpg
```

The prediction script returns the top 5 predicted breeds with their model confidence scores.

### Start the API

Install the API dependencies if needed:

```powershell
pip install -r requirements.txt
```

Start the service:

```powershell
uvicorn api:app --reload
```

Open the interactive API documentation at `http://127.0.0.1:8000/docs`.

The prediction endpoint accepts an image as multipart form data:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@st.bern.jpg"
```

Example response:

```json
{
  "filename": "st.bern.jpg",
  "predicted_breed": "Saint Bernard",
  "confidence": 0.9888,
  "top_predictions": [
    {"breed": "Saint Bernard", "confidence": 0.9888}
  ]
}
```

### Run with Docker

The Docker image is CPU-capable and does not include the dataset. The checkpoint must be present in the project directory before building:

```powershell
docker build -t pet-breed-api .
docker run --rm -p 8000:8000 pet-breed-api
```

Then open `http://127.0.0.1:8000/docs`. GPU support inside Docker on Windows is not assumed or tested by this project.

### Run API tests

The API contract tests use a lightweight mocked model, so they do not retrain the model or require the checkpoint for each test:

```powershell
pytest tests
```

Manual testing with `st.bern.jpg` and `my_pet.jpg` has also verified real checkpoint inference through the API.

## Example

For an input image, the model produces results such as:

```text
Top 5 predictions:

1. Saint Bernard: 98.88%
2. Boxer: 0.73%
3. American Bulldog: 0.15%
4. Leonberger: 0.05%
5. Basset Hound: 0.05%
```

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow
- Scikit-learn
- ResNet18
- CUDA

## Future Improvements

Possible improvements include:

- Training for more epochs
- Experimenting with different learning rates
- Fine-tuning additional ResNet layers
- Comparing ResNet18 with other architectures
- Improving classification of visually similar breeds
- Creating a simple web interface for image prediction
- Adding automated API tests
- Improving Docker image size and deployment configuration
