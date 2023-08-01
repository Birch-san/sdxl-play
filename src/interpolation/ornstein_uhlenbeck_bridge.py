import torch
def ornstein_uhlenbeck_bridge(
  n: int,
  start: torch.FloatTensor,
  end: torch.FloatTensor,
  q=1.0,
):
  """
  Author: Katherine Crowson
  https://discord.com/channels/729741769192767510/730484623028519072/1136013263482671245
  Ornstein-Uhlenbeck bridge
  lacks a KLMC2 like momentum/smoothness
  works for diffusion starting noise (maintains variance throughout) if you set q kind of low, like 0.1, probably
  """
  x = start
  xs = [x]
  ts = torch.linspace(0, 1, n - 1, device=start.device)
  h = ts[1] - ts[0]
  for t in ts[:-1]:
    fac = q * (1 - t)
    dx = q * (-x / torch.tanh(fac) + end / torch.sinh(fac))
    x = x + h * dx
    x = x + torch.sqrt(2 * h) * torch.randn_like(x)
    xs.append(x)
  xs.append(end)
  return torch.stack(xs)