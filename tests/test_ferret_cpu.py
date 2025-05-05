import os
import sys
import unittest
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ferret_model import FerretModel
from src.utils.weight_manager import setup_model_weights, WEIGHTS_DIR

class TestFerretCPU(unittest.TestCase):
    """Test Ferret model execution on CPU."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test case by ensuring model weights are available."""
        cls.test_image_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data')))
        os.makedirs(cls.test_image_dir, exist_ok=True)
        
        cls.test_image_url = "https://raw.githubusercontent.com/apple/ml-ferret/main/ferret/serve/examples/extreme_ironing.jpg"
        
        setup_model_weights(use_7b=True)
    
    def test_ferret_model_initialization(self):
        """Test that the Ferret model can be initialized on CPU."""
        model = FerretModel(device="cpu")
        self.assertIsNotNone(model)
    
    def test_ferret_model_generation(self):
        """Test that the Ferret model can generate responses on CPU."""
        model = FerretModel(device="cpu")
        
        prompt = "What's happening in this image?"
        response = model.generate_response(
            prompt=prompt,
            image_path=self.test_image_url,
            max_new_tokens=100  # Reduced for faster testing
        )
        
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 10)  # Response should be non-trivial

if __name__ == "__main__":
    unittest.main()
