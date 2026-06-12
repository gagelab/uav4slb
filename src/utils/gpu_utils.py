"""
GPU Utility Functions for Automatic GPU Selection

Provides functions to automatically detect and select the best available GPU
for training on multi-GPU systems.
"""

import subprocess
import torch
import logging
from typing import Optional, List, Tuple
import xml.etree.ElementTree as ET


logger = logging.getLogger(__name__)


def get_gpu_memory_usage() -> List[Tuple[int, int, int]]:
    """
    Get memory usage for all GPUs using nvidia-smi.
    
    Returns:
        List of tuples (gpu_id, used_memory_mb, total_memory_mb)
        Empty list if nvidia-smi fails
    """
    try:
        # Run nvidia-smi with XML output for easier parsing
        result = subprocess.run(
            ['nvidia-smi', '-q', '-x'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse XML output
        root = ET.fromstring(result.stdout)
        gpu_info = []
        
        for i, gpu in enumerate(root.findall('gpu')):
            # Get memory info
            fb_memory = gpu.find('fb_memory_usage')
            if fb_memory is not None:
                used = int(fb_memory.find('used').text.split()[0])  # Remove 'MiB'
                total = int(fb_memory.find('total').text.split()[0])
                gpu_info.append((i, used, total))
        
        return gpu_info
        
    except (subprocess.CalledProcessError, ET.ParseError, AttributeError) as e:
        logger.warning(f"Failed to get GPU info via nvidia-smi: {e}")
        return []


def get_gpu_utilization() -> List[Tuple[int, float]]:
    """
    Get GPU utilization percentage for all GPUs.
    
    Returns:
        List of tuples (gpu_id, utilization_percent)
        Empty list if nvidia-smi fails
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpu_util = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                gpu_id = int(parts[0].strip())
                util = float(parts[1].strip())
                gpu_util.append((gpu_id, util))
        
        return gpu_util
        
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning(f"Failed to get GPU utilization: {e}")
        return []


def select_best_gpu(
    memory_threshold: float = 0.1,
    utilization_threshold: float = 10.0,
    prefer_empty: bool = True
) -> Optional[int]:
    """
    Automatically select the best available GPU based on memory and utilization.
    
    Args:
        memory_threshold: Minimum fraction of free memory to consider GPU "available" (0-1)
        utilization_threshold: Maximum utilization % to consider GPU "available"
        prefer_empty: If True, prefer GPUs with no memory used; otherwise use least-used
        
    Returns:
        GPU ID of best available GPU, or None if no GPUs available
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.warning("CUDA not available")
        return None
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.warning("No GPUs detected")
        return None
    
    logger.info(f"Detected {num_gpus} GPUs")
    
    # Get memory usage
    memory_info = get_gpu_memory_usage()
    if not memory_info:
        logger.warning("Could not get GPU memory info, falling back to GPU 0")
        return 0
    
    # Get utilization
    utilization_info = get_gpu_utilization()
    util_dict = {gpu_id: util for gpu_id, util in utilization_info}
    
    # Find available GPUs
    available_gpus = []
    for gpu_id, used_mem, total_mem in memory_info:
        free_mem_fraction = (total_mem - used_mem) / total_mem
        utilization = util_dict.get(gpu_id, 100.0)  # Default to high if unknown
        
        logger.info(
            f"GPU {gpu_id}: {used_mem}/{total_mem} MB used "
            f"({free_mem_fraction*100:.1f}% free), "
            f"utilization: {utilization:.1f}%"
        )
        
        # Check if GPU meets availability criteria
        if free_mem_fraction >= memory_threshold and utilization <= utilization_threshold:
            available_gpus.append({
                'id': gpu_id,
                'used_mem': used_mem,
                'free_mem_fraction': free_mem_fraction,
                'utilization': utilization
            })
    
    if not available_gpus:
        logger.warning("No GPUs meet availability criteria, selecting least-used GPU")
        # Fall back to selecting GPU with most free memory
        best_gpu = min(memory_info, key=lambda x: x[1])  # Min used memory
        logger.info(f"Selected GPU {best_gpu[0]} (least memory used)")
        return best_gpu[0]
    
    # Select best GPU from available ones
    if prefer_empty:
        # Prefer completely empty GPUs first
        empty_gpus = [gpu for gpu in available_gpus if gpu['used_mem'] == 0]
        if empty_gpus:
            best_gpu = empty_gpus[0]
            logger.info(f"Selected empty GPU {best_gpu['id']}")
            return best_gpu['id']
    
    # Select GPU with most free memory
    best_gpu = max(available_gpus, key=lambda x: x['free_mem_fraction'])
    logger.info(
        f"Selected GPU {best_gpu['id']} "
        f"({best_gpu['free_mem_fraction']*100:.1f}% free, "
        f"{best_gpu['utilization']:.1f}% utilized)"
    )
    
    return best_gpu['id']


def get_device(
    device_config: Optional[dict] = None,
    auto_select: bool = True
) -> torch.device:
    """
    Get PyTorch device with automatic GPU selection.
    
    Args:
        device_config: Optional device configuration dict with keys:
                      - use_cuda: bool
                      - cuda_device: int (ignored if auto_select=True)
        auto_select: If True, automatically select best GPU
        
    Returns:
        PyTorch device
    """
    # Check if CUDA should be used
    if device_config and not device_config.get('use_cuda', True):
        logger.info("CUDA disabled in config, using CPU")
        return torch.device('cpu')
    
    if not torch.cuda.is_available():
        logger.info("CUDA not available, using CPU")
        return torch.device('cpu')
    
    # Auto-select GPU
    if auto_select:
        gpu_id = select_best_gpu()
        if gpu_id is None:
            logger.info("No suitable GPU found, using CPU")
            return torch.device('cpu')
        
        logger.info(f"Auto-selected GPU {gpu_id}")
        return torch.device(f'cuda:{gpu_id}')
    
    # Use specified GPU
    if device_config and 'cuda_device' in device_config:
        gpu_id = device_config['cuda_device']
        logger.info(f"Using configured GPU {gpu_id}")
        return torch.device(f'cuda:{gpu_id}')
    
    # Default to GPU 0
    logger.info("Using default GPU 0")
    return torch.device('cuda:0')


def print_gpu_status():
    """Print detailed status of all GPUs."""
    print("\n" + "="*70)
    print("GPU Status")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"Total GPUs: {num_gpus}\n")
    
    memory_info = get_gpu_memory_usage()
    utilization_info = get_gpu_utilization()
    util_dict = {gpu_id: util for gpu_id, util in utilization_info}
    
    for gpu_id, used_mem, total_mem in memory_info:
        free_mem = total_mem - used_mem
        free_pct = (free_mem / total_mem) * 100
        util = util_dict.get(gpu_id, -1)
        
        status = "🟢 Available" if free_pct > 90 and util < 10 else "🟡 In Use"
        
        print(f"GPU {gpu_id}: {status}")
        print(f"  Name: {torch.cuda.get_device_name(gpu_id)}")
        print(f"  Memory: {used_mem:,} / {total_mem:,} MB ({free_pct:.1f}% free)")
        if util >= 0:
            print(f"  Utilization: {util:.1f}%")
        print()
    
    print("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    import sys
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    print("\n" + "="*70)
    print("GPU Selection Utility")
    print("="*70)
    
    # Show GPU status
    print_gpu_status()
    
    # Select best GPU
    best_gpu = select_best_gpu()
    if best_gpu is not None:
        print(f"\n✓ Best GPU for training: {best_gpu}")
        device = torch.device(f'cuda:{best_gpu}')
        print(f"  PyTorch device: {device}")
        print(f"  Device name: {torch.cuda.get_device_name(best_gpu)}")
    else:
        print("\n✗ No suitable GPU found")
        sys.exit(1)
