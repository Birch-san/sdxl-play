from contextlib import ContextDecorator
from dataclasses import dataclass
from .device import DeviceType
from typing import Protocol

class DeviceMovable(Protocol):
  def to(self, device: DeviceType): ...

@dataclass
class to_device(ContextDecorator):
  movable: DeviceMovable
  device: DeviceType

  def __enter__(self):
    self.movable.to(self.device)
    return self

  def __exit__(self, *exc):
    self.movable.to('cpu')
    return False
