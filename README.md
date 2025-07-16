### 환경 세팅
open `pyproject.toml` and change following region:
```
[[tool.uv.index]]
name = "pytorch-cu{your_cuda_version}"
url = "https://download.pytorch.org/whl/cu{your_cuda_version}"
explicit = true
```

and execute `uv sync`