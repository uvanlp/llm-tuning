## icl.py
## Construct demos for in-context learning

from typing import Union, List
import numpy as np


class ICL_Demos(object):
    def __init__(self,
                 demos: List[dict] = [],
                 kshot: int = 1,
                 ):
        self.demos = demos
        self.kshot = kshot
        self.N = len(demos)


    def generate(self,
                 method: str = "random",
                 ):
        if self.N <= 0:
            raise ValueError("No demonstrations for ICL")
        # generation function
        subset = None
        if method == "random":
            subset = self._random_sample()
        else:
            raise NotImplementedError(f"Generation method for ICL {method} has not been implemented yet")
        return subset


    def _random_sample(self):
        inds = np.random.choice(self.N, self.kshot, replace=False)
        subset = [self.demos[idx] for idx in list(inds)]
        return subset


if __name__ == '__main__':
    demos = [{'key':1}, {'del':2}, {'rel':3}]
    kshot = 1
    icl = ICL_Demos(demos, kshot)
    print(icl.generate())
