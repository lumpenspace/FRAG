from threading import Lock
from typing import Any, Self, Tuple, Dict, TypeVar, Generic, Type

T = TypeVar("T", bound=Type[Dict[str, Any]])


class SingletonMixin(Generic[T]):
    _instance: Self | None = None
    _lock = Lock()
    _init_args: Tuple[Any, ...] | None = None
    _init_kwargs: Dict[str, Any] | None = None

    def __new__(cls, *args: Tuple[Any, ...], **kwargs: Type[T]) -> Self:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SingletonMixin, cls).__new__(cls)
                    cls._instance.__init__(*args, **kwargs)
                    cls._init_args = args
                    cls._init_kwargs = kwargs
        else:
            # Check if the arguments match the initial ones
            if args != cls._init_args or kwargs != cls._init_kwargs:
                raise ValueError(
                    "Cannot reinitialize a singleton instance with different arguments."
                )
        return cls._instance

    def __init__(self, *args: Tuple[Any, ...], **kwargs: Type[T]) -> None:
        pass  # Initialization logic should be implemented in the subclass

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None
            cls._init_args = None
            cls._init_kwargs = None
