from contextvars import ContextVar

_data_dir: ContextVar[str] = ContextVar("data_dir", default="data")


def set_data_dir(path: str) -> None:
    _data_dir.set(path)


def get_data_dir() -> str:
    return _data_dir.get()