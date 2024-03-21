import numpy as np
from Stencils import stencils, Stencils

class Derivatives:
    """
    Centered finite differences of order 1, 2
    """
    class Derivative:
        def __init__(self,
                     grid_array,
                     num_ghost,
                     array):
            self.actual_size = np.size(grid_array) - 2*num_ghost           
            self.dx = grid_array[1]-grid_array[0]
            self.oodx = 1./self.dx
            self.is_ = num_ghost
            self.ie_ = num_ghost+self.actual_size
            self.derivative = np.zeros_like(grid_array)


        def calculate_derivative(self,
                         array,
                         order,
                         accuracy_order):
            N = self.actual_size+2*self.is_

            stencil = Stencils(order, accuracy_order)


            S_2 = stencil.width//2
            for i, s in enumerate(stencil.stencil):
                self.derivative[self.is_:self.ie_] = self.derivative[self.is_:self.ie_] + s * array[self.is_ + i-S_2 : self.ie_ + i-S_2]
            self.derivative = self.derivative * self.oodx**order
            self.derivative[0:self.is_] = np.nan
            self.derivative[self.ie_:N] = np.nan

            return

        def get_extremes(self):
            return self.is_, self.ie_

    def __init__(self,
                 order = 1,
                 accuracy_order = 2):
        self.order = order
        self.accuracy_order = accuracy_order
        self.num_ghost = accuracy_order//2



    def __str__(self):
        return f"Derivative of order {self.order} and order of accuracy {self.accuracy_order}."
    
    def calculate_derivative(self,
                            grid_array,
                            array):
        derivative = self.Derivative(grid_array, self.num_ghost, array)
        derivative.calculate_derivative(array,self.order,self.accuracy_order)

        return derivative


if __name__ == "__main__":
    ex = Derivatives()
    print(ex)

