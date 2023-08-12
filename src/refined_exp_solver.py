import torch
from torch import no_grad, FloatTensor
from tqdm import tqdm
from itertools import pairwise
from typing import Protocol, Optional, Dict, Any, TypedDict, NamedTuple
import math

class DenoiserModel(Protocol):
  def __call__(self, x: FloatTensor, t: FloatTensor, *args, **kwargs) -> FloatTensor: ...

class RefinedExpCallbackPayload(TypedDict):
  x: FloatTensor
  i: int
  sigma: FloatTensor
  sigma_hat: FloatTensor

class RefinedExpCallback(Protocol):
  def __call__(self, payload: RefinedExpCallbackPayload) -> None: ...

class NoiseSampler(Protocol):
  def __call__(self, x: FloatTensor) -> FloatTensor: ...

class StepOutput(NamedTuple):
  x_next: FloatTensor
  denoised: FloatTensor
  denoised2: FloatTensor

def gamma(
  n: int,
) -> int:
  """
  https://en.wikipedia.org/wiki/Gamma_function
  for every positive integer n,
  Γ(n) = (n-1)!
  """
  return math.factorial(n-1)

def incomplete_gamma(
  s: int,
  x: float
) -> float:
  """
  https://en.wikipedia.org/wiki/Incomplete_gamma_function
  if s is a positive integer,
  Γ(s, x) = (j-1)!*∑{k=0..s-1}(x^k/k!)
  """
  incomp_gamma_sum: float = 0
  # {k=0..s-1} inclusive
  for k in range(s):
    numerator: float = x**k
    denom: int = math.factorial(k)
    incom_gamma: float = numerator/denom
    incomp_gamma_sum += incom_gamma
  return incomp_gamma_sum

# by Katherine Crowson
def phi_1(neg_h: float):
  return torch.nan_to_num(torch.expm1(neg_h) / neg_h, nan=1.0)

# by Katherine Crowson
def phi_2(neg_h: float):
  return torch.nan_to_num((torch.expm1(neg_h) - neg_h) / neg_h**2, nan=0.5)

# by Katherine Crowson
def phi_3(neg_h: float):
  return torch.nan_to_num((torch.expm1(neg_h) - neg_h - neg_h**2 / 2) / neg_h**3, nan=1 / 6)

def phi(
  neg_h: float,
  j: int,
):
  """
  DEPRECATED: prefer Kat's phi_1, phi_2, phi_3 for now

  Lemma 1
  https://arxiv.org/abs/2308.02157
  ϕj(-h) = 1/h^j*∫{0..h}(e^(τ-h)*(τ^(j-1))/((j-1)!)dτ)

  https://www.wolframalpha.com/input?i=integrate+e%5E%28%CF%84-h%29*%28%CF%84%5E%28j-1%29%2F%28j-1%29%21%29d%CF%84
  = 1/h^j*[(e^(-h)*(-τ)^(-j)*τ(j))/((j-1)!)]{0..h}
  https://www.wolframalpha.com/input?i=integrate+e%5E%28%CF%84-h%29*%28%CF%84%5E%28j-1%29%2F%28j-1%29%21%29d%CF%84+between+0+and+h
  = 1/h^j*((e^(-h)*(-h)^(-j)*h^j*(Γ(j)-Γ(j,-h)))/(j-1)!)
  = (e^(-h)*(-h)^(-j)*h^j*(Γ(j)-Γ(j,-h))/((j-1)!*h^j)
  = (e^(-h)*(-h)^(-j)*(Γ(j)-Γ(j,-h))/(j-1)!
  = (e^(-h)*(-h)^(-j)*(Γ(j)-Γ(j,-h))/Γ(j)
  = (e^(-h)*(-h)^(-j)*(1-Γ(j,-h)/Γ(j))

  requires j>0
  """
  assert j > 0
  gamma_: float = gamma(j)
  incomp_gamma_: float = incomplete_gamma(j, neg_h)

  phi_: float = math.exp(neg_h) * neg_h**-j * (1-incomp_gamma_/gamma_)

  return phi_

class RESDECoeffsSecondOrder(NamedTuple):
  a2_1: float
  b1: float
  b2: float

def de_second_order(
  h: float,
  c2: float
) -> RESDECoeffsSecondOrder:
  """
  Table 3
  https://arxiv.org/abs/2308.02157
  ϕi,j := ϕi,j(-h) = ϕi(-cj*h)
  a2_1 = c2ϕ1,2
       = c2ϕ1(-c2*h)
  b1 = ϕ1 - ϕ2/c2
  """
  # a2_1: float = c2 * phi(j=1, neg_h=-c2*h)
  # phi1: float = phi(j=1, neg_h=-h)
  # phi2: float = phi(j=2, neg_h=-h)
  a2_1: float = c2 * phi_1(-c2*h)
  phi1: float = phi_1(-h)
  phi2: float = phi_2(-h)
  phi2_c2: float = phi2/c2
  b1: float = phi1 - phi2_c2
  b2: float = phi2_c2
  return RESDECoeffsSecondOrder(
    a2_1=a2_1,
    b1=b1,
    b2=b2,
  )  

def refined_exp_sosu_step(
  model: FloatTensor,
  x: FloatTensor,
  sigma: FloatTensor,
  sigma_next: FloatTensor,
  extra_args: Dict[str, Any] = {},
) -> StepOutput:
  """
  Algorithm 1 "RES Second order Single Update Step with c2"
  https://arxiv.org/abs/2308.02157
  """
  lam_next, lam = (s.log().neg() for s in (sigma_next, sigma))
  h: float = lam_next - lam
  c2 = 0.5
  a2_1, b1, b2 = de_second_order(h=h, c2=c2)
  
  denoised: FloatTensor = model(x, lam, **extra_args)

  c2_h: float = c2*h

  x_2: FloatTensor = math.exp(-c2_h)*x + a2_1*h*denoised
  lam_2: float = lam + c2_h

  denoised2: FloatTensor = model(x_2, lam_2, **extra_args)

  x_next: FloatTensor = math.exp(-h)*x + h*(b1*denoised + b2*denoised2)

  assert sum((denoised.isnan().any().item(), denoised.isinf().any().item(), denoised2.isnan().any().item(), denoised2.isinf().any().item(), x_next.isnan().any().item(), x_next.isinf().any().item())) == 0
  
  return StepOutput(
    x_next=x_next,
    denoised=denoised,
    denoised2=denoised2,
  )
  

@no_grad()
def sample_refined_exp_s(
  model: FloatTensor,
  x: FloatTensor,
  sigmas: FloatTensor,
  denoise_to_zero: bool = False,
  extra_args: Dict[str, Any] = {},
  callback: Optional[RefinedExpCallback] = None,
  disable: Optional[bool] = None,
  # degree of stochasticity, η, from 0 to len(sigmas)
  ita = 0.,
  noise_sampler: NoiseSampler = torch.randn_like,
):
  """
  Refined Exponential Solver (S).
  Algorithm 2 "RES Single-Step Sampler"
  https://arxiv.org/abs/2308.02157
  """
  assert sigmas[-1] == 0
  for i, (sigma, sigma_next) in tqdm(enumerate(pairwise(sigmas[:-1])), disable=disable, total=len(sigmas)-2):
    eps: FloatTensor = noise_sampler(x)
    sigma_hat = sigma * (1 + ita)
    x_hat = x + (sigma_hat ** 2 - sigma ** 2) ** .5 * eps
    x_next, denoised, denoised2 = refined_exp_sosu_step(model, x_hat, sigma_hat, sigma_next)
    if callback is not None:
      payload = RefinedExpCallbackPayload(
        x=x,
        i=i,
        sigma=sigma,
        sigma_hat=sigma_hat,
        denoised=denoised,
        denoised2=denoised2,
      )
      callback(payload)
    x = x_next
  if denoise_to_zero:
    eps: FloatTensor = noise_sampler(x)
    sigma_hat = sigma * (1 + ita)
    x_hat = x + (sigma_hat ** 2 - sigma ** 2) ** .5 * eps
    lam: float = x_hat.log().neg()
    x_next: FloatTensor = model(x_hat, lam)
    x = x_next
  return x