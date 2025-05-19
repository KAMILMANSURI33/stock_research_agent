from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, name: str):
        self.name = name
        self._state: Dict[str, Any] = {}
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process the input data and return results."""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the agent."""
        return self._state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the state of the agent."""
        self._state = state
    
    def clear_state(self) -> None:
        """Clear the agent's state."""
        self._state = {}
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent with necessary setup."""
        pass
    
    async def cleanup(self) -> None:
        """Clean up any resources used by the agent."""
        self.clear_state()
    
    def __str__(self) -> str:
        return f"{self.name} Agent" 