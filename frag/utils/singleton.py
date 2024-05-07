from threading import Lock
from typing import Any, Self, Tuple, Dict, TypeVar, Generic, Type

T = TypeVar("T", bound=Type[Dict[str, Any]])


class SingletonMixin(Generic[T]):
    _instance: Dict[str, Self] = {}
    _lock: Dict[str, Lock] = {}
    _init_args: Dict[str, Tuple[Any, ...]] = {}
    _init_kwargs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    @property
    def instance(cls) -> Self | None:
        return cls._instance.get(cls.__name__, None)

    def __new__(cls, *args: Tuple[Any, ...], **kwargs: Type[T]) -> Self:
        name: str = cls.__name__
        if name not in cls._instance:
            if name not in cls._lock:
                cls._lock[name] = Lock()
            with cls._lock[name]:
                if name not in cls._instance:
                    cls._instance[name] = super(SingletonMixin, cls).__new__(cls)
                    cls._instance[name].__init__(*args, **kwargs)
                    cls._init_args[name] = args
                    cls._init_kwargs[name] = kwargs
                return cls._instance[name]
        else:
            # Check if the arguments match the initial ones
            if args != cls._init_args[name] or kwargs != cls._init_kwargs[name]:
                raise ValueError(
                    "Cannot reinitialize a singleton instance with different arguments."
                )
            return cls._instance[name]

    def __init__(self, *args: Tuple[Any, ...], **kwargs: Type[T]) -> None:
        pass  # Initialization logic should be implemented in the subclass

    @classmethod
    def reset(cls) -> None:
        if cls.instance is not None:
            with cls._lock[cls.__name__]:
                cls._instance.pop(cls.__name__)

            cls._init_args.pop(cls.__name__)
            cls._init_kwargs.pop(cls.__name__)
