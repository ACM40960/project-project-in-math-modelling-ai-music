import numpy, torch, sys
print('numpy:', numpy.__version__)
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('numpy_path:', numpy.__file__)
print('python:', sys.executable)
