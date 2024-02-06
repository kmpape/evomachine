from abc import ABC, abstractmethod

class AbstractStrategy(ABC):
    
    @abstractmethod
    def callback(self, fov_id: int, t: int, data: dict) -> None:
        """Callback function for the strategy. This function is called by the\\
        automaton when new data is available. 
        

        Parameters
        ----------
        `fov_id` : int
            The id of the field of view.
        `t` : int
            The time of the data.
        `data` : dict
            Processed image data such as cell positions.
        """
        pass

    @abstractmethod
    def initialise(self) -> None:
        """Initialise the strategy.
        """
        pass

class NoStrategy(AbstractStrategy):
    """Strategy that does nothing.
    """

    def callback(self, fov_id: int, t: int, data: dict) -> None:
        pass

    def initialise(self) -> None:
        pass

class DummyStrategy(AbstractStrategy):
    """Dummy strategy for testing purposes.
    """
    def callback(self, fov_id: int, t: int, data: dict) -> None:
        print(f"FOV {fov_id} at time {t} with data {data}")

