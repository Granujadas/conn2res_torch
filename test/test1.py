import numpy as np
import torch
import time

# -------------------------
# Simulación de función NumPy
def funcion_numpy(x):
    return np.tanh(np.dot(x, x.T))

# Simulación de función PyTorch
def funcion_torch(x):
    return torch.tanh(torch.matmul(x, x.T))

# -------------------------
# Datos de entrada
size = 1000
x_np = np.random.rand(size, size).astype(np.float32)
x_torch = torch.tensor(x_np).cuda()

# -------------------------
# Ejecución NumPy (CPU)
start = time.time()
out_np = funcion_numpy(x_np)
end = time.time()
print(f"NumPy (CPU): {end - start:.4f} segundos")

# -------------------------
# Ejecución PyTorch (GPU)
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    out_torch = funcion_torch(x_torch)
torch.cuda.synchronize()
end = time.time()
print(f"PyTorch (GPU): {end - start:.4f} segundos")

# -------------------------
# Validación de resultados
out_torch_cpu = out_torch.cpu().numpy()
print("¿Resultados similares?", np.allclose(out_np, out_torch_cpu, atol=1e-5))
