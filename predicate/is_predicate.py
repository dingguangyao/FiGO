class IsPredicate:
    def __init__(self, cls, nagetion=False):
        self._cls = cls
        self._nagetion = nagetion

    @property
    def cls(self):
        return self._cls
    
    def evaluate(self, res):
        if self._nagetion == True:
            if res != self._cls:
                return True
            else:
                return False
        else:
            if res == self._cls:
                return True
            else:
                return False
        