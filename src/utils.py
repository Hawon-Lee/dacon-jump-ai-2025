import importlib
from pathlib import Path

def load_model_instance(model_path: str, model_name: str, params: dict):
    """모델 스크립트의 경로와 모델 이름을 받아서 동적으로 모델 import하는 함수"""
    path = Path(model_path)
    # 'src/models/model.py' -> 'src.models.model'
    module_name = ".".join(path.with_suffix("").parts)

    try:
        module = importlib.import_module(module_name)
        class_ = getattr(module, model_name)
        instance = class_(**params)
        print(f"Successfully loaded class '{model_name}' from '{module_name}'")
        return instance
    except ImportError as e:
        raise ImportError(
            f"Could not import module '{module_name}'. Check the path."
        ) from e
    except AttributeError:
        raise AttributeError(
            f"Could not find class '{model_name}' in module '{module_name}'."
        )