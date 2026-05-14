from abc import ABC, abstractmethod
from typing import Annotated

HardwareControllerNameType = Annotated[str, "Unique hardware controller name"]

class HardwareController(ABC):
    
    def __init__(self, name: HardwareControllerNameType):
        self.name: HardwareControllerNameType = name
        self._is_initialised: bool = False
        self._is_alive: bool = False
    
    @abstractmethod
    def initialise(self, force: bool = False) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def reinitialise(self, force: bool = False) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def is_alive(self) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def shutdown(self, force: bool = False) -> None:
        raise NotImplementedError


class Peripheral(ABC):    
    @abstractmethod
    def is_alive(self) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def initialise(self, force: bool = False) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def finalise(self, force: bool = False) -> None:
        raise NotImplementedError


class Stage(Peripheral):
    @abstractmethod
    def get_coordinates(self, axes: list[str]) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def move_to(self, coordinates: dict[str, float], block: bool = True) -> None:
        raise NotImplementedError

    @abstractmethod
    def halt(self) -> None:
        raise NotImplementedError
    

