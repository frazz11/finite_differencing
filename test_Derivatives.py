import pytest
from pytest import approx
import numpy as np 
from Derivatives import Derivatives

def test_numghost1Whenorder1andaccuracyorder2():
    der_test = Derivatives(order = 1,
                          accuracy_order = 2)
    assert der_test.num_ghost == 1

def test_numghost2Whenorder1andaccuracyorder4():
    der_test = Derivatives(order = 1,
                          accuracy_order = 4)
    assert der_test.num_ghost == 2

def test_numghost3Whenorder1andaccuracyorder6():
    der_test = Derivatives(order = 1,
                          accuracy_order = 6)
    assert der_test.num_ghost == 3

def test_numghost4Whenorder1andaccuracyorder8():
    der_test = Derivatives(order = 1,
                          accuracy_order = 8)
    assert der_test.num_ghost == 4

def test_Lencalculate_derivative():
    der_test = Derivatives()
    array_test = np.arange(10)
    der1 = der_test.calculate_derivative(array_test, array_test)
    assert der1.actual_size == 10 - 2*der_test.num_ghost

def test_Derivativedx():
    der_test = Derivatives()
    array_test = np.arange(10)
    der1 = der_test.calculate_derivative(array_test, array_test)
    assert der1.dx == 1

def test_Lencalculate_derivativeEqualLenie_is_():
    der_test = Derivatives()
    array_test = np.arange(10)
    der1 = der_test.calculate_derivative(array_test, array_test)
    assert der1.actual_size == der1.ie_-der1.is_

def test_DerConstAllZeros():
    der_test = Derivatives()
    array_grid_test = np.arange(10)
    array_test = np.empty(10)
    array_test.fill(10)
    der1 = der_test.calculate_derivative(array_grid_test, array_test)
    assert np.allclose(der1.derivative[der1.is_:der1.ie_], np.zeros(der1.actual_size))

def test_DerConstAllZerosFloat():
    der_test = Derivatives()
    array_grid_test = np.arange(-1,1,0.1)
    array_test = np.empty(np.size(array_grid_test))
    array_test.fill(10)
    der1 = der_test.calculate_derivative(array_grid_test, array_test)
    assert np.allclose(der1.derivative[der1.is_:der1.ie_], np.zeros(der1.actual_size))

def test_Der2xEqual2Int():
    der_test = Derivatives(1,2)
    array_grid_test = np.arange(10)
    array_test = np.copy(array_grid_test)*2
    der1 = der_test.calculate_derivative(array_grid_test, array_test)
    assert np.allclose(der1.derivative[der1.is_:der1.ie_], np.ones(der1.actual_size)*2)

def test_Der2xEqual2Float():
    der_test = Derivatives(1,2)
    array_grid_test = np.arange(-1,1,0.1)
    array_test = np.copy(array_grid_test)*2
    der1 = der_test.calculate_derivative(array_grid_test, array_test)
    assert np.allclose(der1.derivative[der1.is_:der1.ie_], np.ones(der1.actual_size)*2)

def test_DerxsquaredEqual2x():
    der_test = Derivatives(1,2)
    array_grid_test = np.arange(-1,1,0.1)
    array_test = np.copy(array_grid_test**2)
    der1 = der_test.calculate_derivative(array_grid_test, array_test)
    assert np.allclose(der1.derivative[der1.is_:der1.ie_], array_grid_test[der1.is_:der1.ie_]*2)

def test_DerxsquaredEqual2x_4thorder():
    der_test = Derivatives(1,4)
    array_grid_test = np.arange(-1,1,0.1)
    array_test = np.copy(array_grid_test**2)
    der1 = der_test.calculate_derivative(array_grid_test, array_test)
    assert np.allclose(der1.derivative[der1.is_:der1.ie_], array_grid_test[der1.is_:der1.ie_]*2)

def test_DerxsquaredEqual2x_6thorder():
    der_test = Derivatives(1,6)
    array_grid_test = np.arange(-1,1,0.1)
    array_test = np.copy(array_grid_test**2)
    der1 = der_test.calculate_derivative(array_grid_test, array_test)
    assert np.allclose(der1.derivative[der1.is_:der1.ie_], array_grid_test[der1.is_:der1.ie_]*2)

def test_SecDerxsquaredEqual2_2ndorder():
    der_test = Derivatives(2,2)
    array_grid_test = np.arange(-1,1,0.1)
    array_test = np.copy(array_grid_test**2)
    der1 = der_test.calculate_derivative(array_grid_test, array_test)
    assert np.allclose(der1.derivative[der1.is_:der1.ie_], np.ones(der1.ie_-der1.is_)*2)

def test_SecDerxsquaredEqual2_4thorder():
    der_test = Derivatives(2,4)
    array_grid_test = np.arange(-1,1,0.1)
    array_test = np.copy(array_grid_test**2)
    der1 = der_test.calculate_derivative(array_grid_test, array_test)
    assert np.allclose(der1.derivative[der1.is_:der1.ie_], np.ones(der1.ie_-der1.is_)*2)

@pytest.fixture(params=[2,4,6])
def ConvergenceDer1(request):
    accuracy = request.param
    dx = 1
    der_test = Derivatives(1,accuracy)
    max_error = []
    step_size = []

    for i in range(20):
        dx = dx/2
        array_grid_test = np.arange(-4,4,dx)
        function = np.cos(array_grid_test)
        der = der_test.calculate_derivative(array_grid_test, function)
        exact_derivative = -np.sin(array_grid_test)

        max_error_ = np.max(np.abs(der.derivative[der.is_:der.ie_]-exact_derivative[der.is_:der.ie_]))

        if not len(max_error) == 0:
            if max_error_ < max_error[-1]:
                pass
            else:
                break
        max_error.append(max_error_)
        step_size.append(dx)
        del der

    max_error.pop()
    step_size.pop()

    step_size = np.array(step_size)
    max_error = np.array(max_error)

    mask = max_error > 1e-10
    m, q = np.polyfit(np.log(step_size[mask]), np.log(max_error[mask]), 1)

    return m == approx(accuracy, abs = 1e-1)

def test_Der1(ConvergenceDer1):
    assert ConvergenceDer1

@pytest.fixture(params=[2,4,6])
def ConvergenceDer2(request):
    accuracy = request.param
    dx = 1
    der_test = Derivatives(2,accuracy)
    max_error = []
    step_size = []

    for i in range(20):
        dx = dx/2
        array_grid_test = np.arange(-4,4,dx)
        function = np.cos(array_grid_test)
        der = der_test.calculate_derivative(array_grid_test, function)
        exact_derivative = -np.cos(array_grid_test)

        max_error_ = np.max(np.abs(der.derivative[der.is_:der.ie_]-exact_derivative[der.is_:der.ie_]))

        if not len(max_error) == 0:
            if max_error_ < max_error[-1]:
                pass
            else:
                break
        max_error.append(max_error_)
        step_size.append(dx)
        del der

    max_error.pop()
    step_size.pop()

    step_size = np.array(step_size)
    max_error = np.array(max_error)

    mask = max_error > 1e-10
    m, q = np.polyfit(np.log(step_size[mask]), np.log(max_error[mask]), 1)

    return m == approx(accuracy, abs = 1e-1)

def test_Der2(ConvergenceDer2):
    assert ConvergenceDer2
