{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy.stats import norm, binom\n",
    "from typing import Callable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Payoff functions\n",
    "def call_payoff(spot, strike):\n",
    "    return np.maximum(spot - strike, 0.0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "def put_payoff(spot, strike):\n",
    "    return np.maximum(strike - spot, 0.0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Price functions\n",
    "def european_binomial_pricer(spot: float, strike: float, expiry: float, rate: float, div: float, vol: float, num: int, payoff: Callable) -> float:\n",
    "    nodes = num + 1\n",
    "    h = expiry / num\n",
    "    u = np.exp((rate - div) * h + vol * np.sqrt(h))\n",
    "    d = np.exp((rate - div) * h - vol * np.sqrt(h))\n",
    "    disc = np.exp(-rate * expiry)\n",
    "    pu = (np.exp((rate - div)*h) - d) / (u - d)\n",
    "\n",
    "    spot_t = 0.0\n",
    "    payoff_t = 0.0\n",
    "\n",
    "    for i in range(nodes):\n",
    "        spot_t = spot * (u ** (num - i)) * (d ** i)\n",
    "        payoff_t +=  payoff(spot_t, strike) * binom.pmf(num - i, num, pu)\n",
    "\n",
    "    return disc * payoff_t"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "def american_binomial_pricer(spot: float, strike: float, expiry: float, rate: float, div: float, vol: float, num: int, payoff: Callable) -> float:\n",
    "    nodes = num + 1\n",
    "    h = expiry / num \n",
    "    u = np.exp(((rate - div) * h) + vol * np.sqrt(h)) \n",
    "    d = np.exp(((rate - div) * h) - vol * np.sqrt(h))\n",
    "    pu = (np.exp((rate - div) * h) - d) / (u - d)\n",
    "    pd = 1 - pu\n",
    "    disc = np.exp(-rate * h)\n",
    "    dpu = disc * pu\n",
    "    dpd = disc * pd\n",
    "\n",
    "    Ct = np.zeros(nodes)\n",
    "    St = np.zeros(nodes)\n",
    "\n",
    "    for i in range(nodes):\n",
    "        St[i] = spot * (u ** (num - i)) * (d ** i)\n",
    "        Ct[i] = payoff(St[i], strike)\n",
    "\n",
    "    for i in range((num - 1), -1, -1):\n",
    "        for j in range(i+1):\n",
    "            Ct[j] = dpu * Ct[j] + dpd * Ct[j+1]\n",
    "            St[j] = St[j] / u\n",
    "            Ct[j] = np.maximum(Ct[j], payoff(St[j], strike))\n",
    "\n",
    "    return Ct[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "def black_scholes_call(spot: float, strike: float, expiry: float, rate: float, div: float, vol: float) -> float:\n",
    "    d1 = (np.log(spot/strike) + (rate - div + 0.5 * vol * vol) * expiry) / (vol * np.sqrt(expiry))\n",
    "    d2 = d1 - vol * np.sqrt(expiry) \n",
    "    return (spot * np.exp(-div * expiry) * norm.cdf(d1)) - (strike * np.exp(-rate * expiry) * norm.cdf(d2))\n",
    "\n",
    "def black_scholes_put(spot: float, strike: float, expiry: float, rate: float, div: float, vol: float) -> float:\n",
    "    d1 = (np.log(spot/strike) + (rate - div + 0.5 * vol * vol) * expiry) / (vol * np.sqrt(expiry))\n",
    "    d2 = d1 - vol * np.sqrt(expiry) \n",
    "    return (strike * np.exp(-rate * expiry) * norm.cdf(-d2)) - (spot * np.exp(-div * expiry) * norm.cdf(-d1))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Delta \n",
    "def binomial_delta(spot: float, strike: float, expiry: float, rate: float, div: float, vol: float, num: int, payoff: Callable) -> float:\n",
    "    h = expiry / num\n",
    "    u = np.exp((rate - div) * h + np.sqrt(h) * vol)\n",
    "    d = np.exp((rate - div) * h - np.sqrt(h) * vol)\n",
    "    Su = u * u * spot\n",
    "    Sd = d * d * spot\n",
    "    Fu = payoff(Su, strike) \n",
    "    Fd = payoff(Sd, strike)\n",
    "    return (Fu - Fd) / (Su - Sd)\n",
    "\n",
    "def simple_delta(spot: float, strike: float, expiry: float, rate: float, div: float, vol: float, num: int, payoff: Callable) -> float:\n",
    "    h = expiry / num\n",
    "    u = np.exp((rate - div) * h + np.sqrt(h) * vol)\n",
    "    d = np.exp((rate - div) * h - np.sqrt(h) * vol)\n",
    "    Su =  u * spot\n",
    "    Sd =  d * spot\n",
    "    Fu = payoff(Su, strike) \n",
    "    Fd = payoff(Sd, strike)\n",
    "    return (Fu - Fd) / (Su - Sd)\n",
    "\n",
    "def black_scholes_call_delta(spot: float, strike: float, tau: float, rate: float, div: float, vol: float) -> float:\n",
    "    d1 = (np.log(spot/strike) + (rate - div + 0.5 * vol * vol) * tau) / (vol * np.sqrt(tau))\n",
    "    return np.exp(-div * tau) * norm.cdf(d1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Simulations\n",
    "def binomial_path(spot: float, expiry: float, rate: float, div: float, vol: float, num: int) -> np.ndarray:\n",
    "    h = expiry / num\n",
    "    u = np.exp((rate - div) * h + np.sqrt(h) * vol)\n",
    "    d = np.exp((rate - div) * h - np.sqrt(h) * vol)\n",
    "    pstar = (np.exp((rate - div) * h) - d) / (u - d) \n",
    "    z = np.random.uniform(size=num)\n",
    "    path = np.empty(num)\n",
    "    path[0] = spot\n",
    "\n",
    "    for i in range(1, num):\n",
    "        if z[i] >= pstar: path[i] = u * path[i-1]\n",
    "        else: z[i] = path[i] = d * path[i-1]\n",
    "\n",
    "    return path\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
