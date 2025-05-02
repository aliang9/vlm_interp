import os
import shutil
import subprocess
from pathlib import Path
import requests
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/weights')))
VICUNA_DIR = WEIGHTS_DIR / "vicuna-7b-v1-3"
FERRET_DIR = WEIGHTS_DIR / "ferret-7b-v1-3"
DELTA_DIR = WEIGHTS_DIR / "ferret-7b-delta"
PROJECTOR_DIR = WEIGHTS_DIR / "projector-7b"

DELTA_URL = "https://docs-assets.developer.apple.com/ml-research/models/ferret/ferret-7b-delta.tar.gz"
PROJECTOR_URL = "https://huggingface.co/liuhaotian/llava-336px-pretrain-vicuna-7b-v1-3/resolve/main/mm_projector.bin"

def ensure_dir(dir_path):
    """Ensure directory exists."""
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def download_file(url, dest_path):
    """Download a file from URL to destination path with progress bar."""
    logger.info(f"Downloading {url} to {dest_path}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    
    return dest_path

def extract_tarball(tarball_path, extract_dir):
    """Extract a tarball to the specified directory."""
    logger.info(f"Extracting {tarball_path} to {extract_dir}")
    ensure_dir(extract_dir)
    
    subprocess.run(['tar', '-xzf', tarball_path, '-C', extract_dir], check=True)
    
    return extract_dir

def apply_delta_weights(base_dir, delta_dir, target_dir):
    """Apply delta weights to base model."""
    logger.info(f"Applying delta weights from {delta_dir} to {base_dir} -> {target_dir}")
    
    if not os.path.exists(delta_dir):
        raise FileNotFoundError(f"Delta weights directory {delta_dir} not found")
    
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base model directory {base_dir} not found")
    
    ensure_dir(target_dir)
    
    cmd = [
        "python", "-c",
        "from transformers.models.llama.modeling_llama import LlamaForCausalLM; "
        "import torch; "
        f"base = LlamaForCausalLM.from_pretrained('{base_dir}'); "
        f"delta = torch.load('{delta_dir}/pytorch_model.bin'); "
        "for key in delta.keys(): "
        "    if key in base.state_dict(): "
        "        base.state_dict()[key] += delta[key]; "
        f"base.save_pretrained('{target_dir}')"
    ]
    
    subprocess.run(cmd, check=True)
    
    for filename in ['tokenizer.model', 'tokenizer_config.json', 'config.json']:
        src_file = os.path.join(base_dir, filename)
        if os.path.exists(src_file):
            shutil.copy(src_file, os.path.join(target_dir, filename))
    
    return target_dir

def setup_model_weights(use_7b=True):
    """Set up model weights by downloading and configuring them."""
    global VICUNA_DIR, FERRET_DIR, DELTA_DIR, PROJECTOR_DIR, DELTA_URL, PROJECTOR_URL
    
    if not use_7b:
        VICUNA_DIR = WEIGHTS_DIR / "vicuna-13b-v1-3"
        FERRET_DIR = WEIGHTS_DIR / "ferret-13b-v1-3"
        DELTA_DIR = WEIGHTS_DIR / "ferret-13b-delta"
        PROJECTOR_DIR = WEIGHTS_DIR / "projector-13b"
        DELTA_URL = "https://docs-assets.developer.apple.com/ml-research/models/ferret/ferret-13b-delta.tar.gz"
        PROJECTOR_URL = "https://huggingface.co/liuhaotian/llava-336px-pretrain-vicuna-13b-v1-3/resolve/main/mm_projector.bin"
    
    ensure_dir(WEIGHTS_DIR)
    
    logger.warning("Vicuna model needs to be downloaded manually due to licensing restrictions.")
    logger.warning(f"Please download Vicuna v1.3 and place it in {VICUNA_DIR}")
    
    delta_tarball = WEIGHTS_DIR / f"ferret-{'7b' if use_7b else '13b'}-delta.tar.gz"
    if not os.path.exists(DELTA_DIR):
        download_file(DELTA_URL, delta_tarball)
        extract_tarball(delta_tarball, DELTA_DIR)
    
    projector_file = PROJECTOR_DIR / "mm_projector.bin"
    ensure_dir(PROJECTOR_DIR)
    if not os.path.exists(projector_file):
        download_file(PROJECTOR_URL, projector_file)
    
    if not os.path.exists(VICUNA_DIR):
        logger.error(f"Vicuna model not found at {VICUNA_DIR}")
        logger.error("Please download manually following instructions from https://github.com/lm-sys/FastChat")
        return False
    
    if not os.path.exists(FERRET_DIR):
        apply_delta_weights(VICUNA_DIR, DELTA_DIR, FERRET_DIR)
    
    return True
