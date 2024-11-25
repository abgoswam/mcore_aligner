import torch

# Example tensor
t1 = torch.randn(1, 18, 200019)  # Shape: [1, 18, 200019]
n = 18

# Reshaping to remove the first dimension
result = t1.view(n, -1)  # Shape: [18, 200019]

# Verify the result
assert result.shape == (n, 200019)
print(result.shape)
