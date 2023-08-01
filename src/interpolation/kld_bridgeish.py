import torch
from typing import Optional
def kld_bridgeish(
  n: int,
  start: torch.FloatTensor,
  end: torch.FloatTensor,
  t_end=1.0,
  u=1.0,
  gamma=2.0,
  q=1.0,
  v=None,
  # note: for some reason CPU Generator resulted in corrupted image
  generator: torch.Generator = torch.Generator(device='cuda'),
) -> Optional[torch.FloatTensor]:
  """
  Author: Katherine Crowson
  SDE that starts at one point and ends at another and has smooth, kind of like kinetic Langevin type motion in the middle
  https://discord.com/channels/729741769192767510/730484623028519072/1136006669579595796
  """
  x = start
  xs = [x]
  ts = torch.linspace(0, t_end, n - 1, device=start.device)
  h = ts[1] - ts[0]
  v = torch.randn_like(x) if v is None else v
  for t in ts[:-1]:
    fac = q * (t_end - t)
    dx = v + q * (-x / torch.tanh(fac) + end / torch.sinh(fac))
    grad = x - torch.lerp(start, end, t / t_end)
    dv = -(gamma * v + u * grad)
    x = x + h * dx
    v = v + h * dv
    # randn_like didn't accept a Generator so I had to get a bit creative
    v = v + torch.sqrt(2 * gamma * u * h) * torch.randn(v.size(), generator=generator, device=generator.device, dtype=v.dtype, layout=v.layout).to(v.device)
    xs.append(x)
  xs.append(end)
  return torch.stack(xs)