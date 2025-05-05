# vlm_interp

A Python package for running the Ferret multimodal language model locally. This implementation is based on [Apple's Ferret model](https://github.com/apple/ml-ferret).

## Features

- Run the Ferret model locally on CPU or GPU
- Load and manage model weights
- Process images and generate text responses
- Support for region-specific queries
- Simple API for integration into other projects

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aliang9/vlm_interp.git
cd vlm_interp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up model weights:
The Ferret model requires Vicuna v1.3 as the base model and LLaVA's first-stage pre-trained projector weight. The setup process will:
- Download Ferret delta weights
- Download LLaVA's projector weights
- Prompt you to download Vicuna v1.3 manually (due to licensing restrictions)
- Apply delta weights to create the final model

```bash
python -m examples.ferret_example --setup_weights --image_path <path_to_image>
```

## Model Weights Storage

Model weights are stored in the `data/weights` directory with the following structure:
- `data/weights/vicuna-7b-v1-3`: Vicuna base model (needs to be downloaded manually)
- `data/weights/ferret-7b-delta`: Ferret delta weights
- `data/weights/ferret-7b-v1-3`: Final Ferret model
- `data/weights/projector-7b`: LLaVA's projector weights

## Usage

### Basic Usage

```python
from src.models.ferret_model import FerretModel

# Initialize model
model = FerretModel(device="cpu")  # Use "cuda" for GPU

# Generate response for an image
response = model.generate_response(
    prompt="Describe this image in detail.",
    image_path="path/to/image.jpg"
)

print(response)
```

### Command Line Example

```bash
python -m examples.ferret_example --image_path path/to/image.jpg --prompt "What's happening in this image?"
```

### Region-Specific Queries

```python
response = model.generate_response(
    prompt="What is in the highlighted region?",
    image_path="path/to/image.jpg",
    region_coords=[100, 100, 300, 300]  # [x1, y1, x2, y2]
)
```

## Testing

Run the test suite to verify CPU execution:

```bash
python -m unittest discover tests
```

## License

This implementation is for research use only and is subject to the license of the original Ferret model (CC BY NC 4.0).
