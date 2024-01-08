class DigPredicate:
    def __init__(self, digital, leq=False):
        self._digital = digital
        self._leq = leq
    
    def evaluate(self, res):
        if self._leq == True:
            if res <= self._digital:
                return True
            else:
                return False
        else:
            if res == self._digital:
                return True
            else:
                return False
        