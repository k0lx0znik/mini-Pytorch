# mini-Pytorch
Implementation of Pytorch by NumPy

`mini-PyTorch` is a lightweight educational re-implementation of core PyTorch functionality.  
It is designed to be **small, transparent, and easy to read**, making it ideal for learning how modern deep-learning frameworks work under the hood.

The project includes:
- A **tensor** class with autograd  
- **Neural network modules** (`Linear`, `ReLU`, etc.)  
- **Optimizers**  
- **Loss functions**  
- **Training utilities**  
- **Extensive tests** proving correctness of individual components  
- A **demo** showing the full workflow  

---

## Features

### Autograd Engine
- Reverse-mode automatic differentiation  
- Computation graph construction  
- Backward pass with gradient propagation  
- Support for basic operations: add, mul, matmul, sum, mean, etc.

### Neural Network Modules
- `Linear`  
- `ReLU`  
- `Sigmoid`  
- `Tanh`  
- `Sequential`  
- Easily extensible with custom modules

### Optimizers
- `SGD`  
- `Adam`

### Loss Functions
- `MSELoss`  
- `CrossEntropyLoss`

### Tests
Located in `tests/`, covering:
- Tensor operations  
- Autograd correctness  
- Module forward/backward  
- Optimizers  
- Loss functions  
