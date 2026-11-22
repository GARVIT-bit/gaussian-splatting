import pytest
import torch


@pytest.fixture
def float64():
    """Run a test with float64 as the default dtype, restoring the previous default after."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)
