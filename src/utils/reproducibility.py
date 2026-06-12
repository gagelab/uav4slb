"""
Reproducibility utilities for UAV SLB project.



This module provides functions to ensure reproducible results across runs
by setting random seeds for various libraries.
"""

import os
import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 48, deterministic: bool = False) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but fully reproducible).
                      Note: Some operations may not have deterministic implementations.
    
    Example:
        >>> set_seed(42)
        >>> # Now all random operations will be reproducible
    """
    # Python random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    if deterministic:
        # Make CuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Use deterministic algorithms (PyTorch 1.8+)
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        
        # Set environment variable for CUBLAS
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        # Enable CuDNN benchmark for better performance (non-deterministic)
        torch.backends.cudnn.benchmark = True


def get_random_state() -> dict:
    """
    Get current random state from all libraries.
    
    Returns:
        Dictionary containing random states
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state()
        if torch.cuda.device_count() > 1:
            state['torch_cuda_all'] = torch.cuda.get_rng_state_all()
    
    return state


def set_random_state(state: dict) -> None:
    """
    Restore random state from saved state dictionary.
    
    Args:
        state: Dictionary containing random states (from get_random_state())
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['torch_cuda'])
    
    if 'torch_cuda_all' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda_all'])


def seed_worker(worker_id: int) -> None:
    """
    Seed function for PyTorch DataLoader workers.
    
    This ensures that each worker has a different but reproducible seed.
    Use with DataLoader's worker_init_fn parameter.
    
    Args:
        worker_id: Worker ID assigned by DataLoader
        
    Example:
        >>> dataloader = DataLoader(
        ...     dataset,
        ...     worker_init_fn=seed_worker,
        ...     generator=torch.Generator().manual_seed(42)
        ... )
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_generator(seed: int) -> torch.Generator:
    """
    Create a PyTorch Generator with a specific seed.
    
    Useful for DataLoader to ensure reproducible data loading.
    
    Args:
        seed: Random seed
        
    Returns:
        PyTorch Generator with seed set
        
    Example:
        >>> generator = create_generator(42)
        >>> dataloader = DataLoader(dataset, generator=generator)
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def make_reproducible(
    seed: int = 42,
    deterministic: bool = False,
    warn: bool = True
) -> dict:
    """
    Comprehensive function to set up reproducibility.
    
    This function:
    1. Sets all random seeds
    2. Configures deterministic behavior if requested
    3. Returns configuration info
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower)
        warn: If True, print warnings about determinism
        
    Returns:
        Dictionary with reproducibility configuration info
        
    Example:
        >>> config = make_reproducible(seed=42, deterministic=True)
    """
    set_seed(seed, deterministic=deterministic)
    
    info = {
        'seed': seed,
        'deterministic': deterministic,
        'cuda_available': torch.cuda.is_available(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
    }
    
    if warn and deterministic:
        print("=" * 80)
        print("WARNING: Deterministic mode enabled")
        print("- Training will be fully reproducible but may be slower")
        print("- Some operations may not have deterministic implementations")
        print("- You may see warnings about non-deterministic operations")
        print("=" * 80)
    
    if warn and not deterministic:
        print("=" * 80)
        print("INFO: Non-deterministic mode (faster, less reproducible)")
        print("- Seeds are set but CuDNN benchmark is enabled")
        print("- Results should be similar but not identical across runs")
        print("=" * 80)
    
    return info


def check_reproducibility(
    model: torch.nn.Module,
    input_size: tuple,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed: int = 42,
    n_runs: int = 3
) -> bool:
    """
    Test if model forward pass is reproducible.
    
    Args:
        model: PyTorch model to test
        input_size: Input tensor size (e.g., (1, 3, 224, 224))
        device: Device to run on
        seed: Seed to use for testing
        n_runs: Number of runs to compare
        
    Returns:
        True if all runs produce identical outputs
        
    Example:
        >>> model = MyModel()
        >>> is_reproducible = check_reproducibility(
        ...     model, 
        ...     input_size=(1, 3, 224, 224)
        ... )
    """
    model = model.to(device)
    model.eval()
    
    outputs = []
    
    for run in range(n_runs):
        # Reset seed and model
        set_seed(seed, deterministic=True)
        
        # Create random input
        x = torch.randn(input_size, device=device)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        outputs.append(output.cpu().numpy())
    
    # Check if all outputs are identical
    for i in range(1, len(outputs)):
        if not np.allclose(outputs[0], outputs[i], rtol=1e-5, atol=1e-8):
            print(f"Run 0 and Run {i} differ!")
            print(f"Max difference: {np.max(np.abs(outputs[0] - outputs[i]))}")
            return False
    
    print(f"Model is reproducible across {n_runs} runs!")
    return True


# Example usage and testing
if __name__ == "__main__":
    print("Testing reproducibility utilities...")
    
    # Test basic seeding
    print("\n1. Testing basic seed setting:")
    set_seed(42)
    random_nums = [random.random() for _ in range(3)]
    print(f"Random numbers (seed=42): {random_nums}")
    
    set_seed(42)
    random_nums_2 = [random.random() for _ in range(3)]
    print(f"Random numbers (seed=42, repeat): {random_nums_2}")
    print(f"Identical: {random_nums == random_nums_2}")
    
    # Test PyTorch seeding
    print("\n2. Testing PyTorch seeding:")
    set_seed(42)
    torch_tensor = torch.randn(3)
    print(f"PyTorch tensor (seed=42): {torch_tensor}")
    
    set_seed(42)
    torch_tensor_2 = torch.randn(3)
    print(f"PyTorch tensor (seed=42, repeat): {torch_tensor_2}")
    print(f"Identical: {torch.allclose(torch_tensor, torch_tensor_2)}")
    
    # Test comprehensive setup
    print("\n3. Testing comprehensive setup:")
    config = make_reproducible(seed=42, deterministic=False, warn=False)
    print(f"Reproducibility config: {config}")
    
    print("\nAll tests completed!")
