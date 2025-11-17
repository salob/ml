import torch
 
if torch.backends.mps.is_available():

    device = torch.device("mps")

    print("Using GPU: Apple Silicon MPS")

else:

    device = torch.device("cpu")

    print("Using CPU")
 
# Example tensor

x = torch.randn(3, 3).to(device)

print(x)