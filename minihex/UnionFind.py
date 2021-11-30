import numpy as np

class Forest:
    def __init__(self, n):
        self.n = n
        self.roots = np.zeros((n + 1,), dtype=np.uint)
        self.ranks = np.zeros((n + 1,), dtype=np.uint)
        for i in range(1, n):
            self.roots[i] = i
            self.ranks[i] = 1

    def in_same_set(self, x, y):
        assert x > 0 and y > 0
        
        if x == y:
            return True
        
        return self.find(x) == self.find(y)

    def find(self, x):
        assert x > 0
        
        R = np.uint(x)
        while R != self.roots[R]:
            R = self.roots[R]

        # path compression
        y = np.uint(x)
        while y != self.roots[y]:
            tmp = self.roots[y]
            self.roots[y] = R
            y = tmp

        return R

    def unite(self, x, y):
        assert x > 0 and y > 0
        
        if x == y:
            return True
        
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return True

        # union by rank
        if self.ranks[x] > self.ranks[y]:
            self.roots[y] = x
            self.ranks[x] += self.ranks[y]
        else:
            self.roots[x] = y
            self.ranks[y] += self.ranks[x]


def tests():
    f = Forest(1000)
    f.unite(1, 2)
    f.unite(3, 4)
    assert f.in_same_set(1, 3) == False
    assert f.in_same_set(1, 2) == True
    f.unite(1, 3)
    assert f.in_same_set(1, 4) == True 

if __name__ == "__main__":
    tests()
