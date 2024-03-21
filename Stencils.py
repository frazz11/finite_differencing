stencils = {
    "1,2": [-1/2, 0.0, 1/2],
    "1,4": [1/12, -2/3, 0, 2/3, -1/12],
    "1,6": [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60],
    "2,2": [1, -2, 1],
    "2,4": [-1/12, 4/3, -5/2, 4/3, -1/12],
    "2,6": [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]
}

class Stencils:
    def __init__(self, order, accuracy_order):
        self.order = order
        self.accuracy_order = accuracy_order
        self.stencil = stencils[f"{order},{accuracy_order}"]
        self.width = len(self.stencil)
