import os
import sys
import argparse
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ferret_model import FerretModel
from src.utils.weight_manager import setup_model_weights

def run_example(args):
    """Run an example of Ferret model on a sample question-image pair."""
    if args.setup_weights:
        setup_model_weights(use_7b=args.use_7b)
    
    model = FerretModel(
        model_path=args.model_path,
        device=args.device,
        use_4bit=args.use_4bit,
        use_7b=args.use_7b
    )
    
    response = model.generate_response(
        prompt=args.prompt,
        image_path=args.image_path,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        region_coords=args.region_coords
    )
    
    print(f"Prompt: {args.prompt}")
    print(f"Response: {response}")
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(f"Prompt: {args.prompt}\n")
            f.write(f"Response: {response}\n")
        print(f"Response saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ferret model on a sample question-image pair")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image file or URL")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.", help="Text prompt")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model weights")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run model on (cpu or cuda)")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--region_coords", type=float, nargs=4, default=None, help="Region coordinates [x1, y1, x2, y2]")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save response")
    parser.add_argument("--setup_weights", action="store_true", help="Setup model weights")
    parser.add_argument("--use_7b", action="store_true", default=True, help="Use 7B model (default)")
    parser.add_argument("--use_13b", dest="use_7b", action="store_false", help="Use 13B model")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    
    args = parser.parse_args()
    run_example(args)
