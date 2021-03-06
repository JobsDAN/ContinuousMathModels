import numpy as np
import scipy.integrate as sp
import scipy.misc as ms
import matplotlib.pyplot as plt
import scipy.optimize as opt

# TODO: Погрешность
## Задача здесь уже сформулирована как (4.18)-(4.19), метод наименьших квадратов

#точное значение функции в t
def exect(t):
	return t**4

### тригонометрические полиномы
def e(k, t):
	return np.sin(np.pi * k * (t - 1))

# вторая производная
def e2(k, t):
	return -(np.pi**2 * k**2 * np.sin(np.pi * k * (t - 1)))

## функция замены
def G(t):
	return 15 * t - 14

## начальное приближение
def y_0(t):
	return 0.0

## значение функции на прошлой итерации
def getYn(coefs):
	if len(coefs) == 0:
		return y_0

	def yn(t):
		accum = 0.0
		for k in range(len(coefs)):
			accum += coefs[k] * e(k + 1, t)
		return accum

	return yn

## оператор А
def operator(k, t):
	return -e2(k, t) + 6 * t**(-10) * (y_0(t) + G(t))**2 * e(k, t)

## интеграл, стоящий слева от знака равенства
def integrate_left(k, j, A, B):
	return sp.quad(lambda t: operator(k, t) * operator(j, t), A, B)[0]

## интеграл, стоящий справа от знака равенства
def integrate_right(j, prevCoefsAk, A, B):
	y_n = getYn(prevCoefsAk)
	return sp.quad(lambda t: (
		6 * t**(-10) * (y_0(t) + G(t))**2 * y_n(t) - 
		2 * t**(-10) * (y_n(t) + G(t))**3 - 10 * t**2) *
			operator(j, t), A, B)[0]

def plotGraph(A, B, stepSize, coefsAk):
	stepCount = int((B - A) / stepSize + 1)
	x_points = np.arange(A, B + stepSize, stepSize)

	exectX = []
	approximateX = []
	for i in range(stepCount):
		t = A + i * stepSize
		exectX.append(exect(t))

		y_n = getYn(coefsAk)
		x_n = y_n(t) + G(t)
		approximateX.append(x_n)

	plt.plot(x_points, exectX, 'b', x_points, approximateX, 'r')
	plt.show()

def solveLSM(prevCoefs, iterCount, A, B):
	iters = range(1, iterCount + 1)

	# вектор правой части СЛАУ
	f = [ integrate_right(j, prevCoefs, A, B) for j in iters ]

	# матрица коэфициентов
	# коэффициенты для каждого уравнения
	coefs = [
	    [ integrate_left(k, j, A, B) for k in iters ]
	    for j in iters
	]

	return np.linalg.solve(coefs, f)

def norm_L2(f, a, b):
	I = sp.quad(lambda t: abs(f(t))**2, a, b)[0]
	return np.square(I)

def diff(a, b):
	def f(t):
		return a(t) - b(t)
	return f

def getXn(coefs):
	def xn(t):
		return getYn(coefs)(t) + G(t)
	return xn

def getOptimalN(eps):
	N = 1
	while True:
		yn0 = getYn(solveLSM([], N + 0, A, B))
		yn1 = getYn(solveLSM([], N + 1, A, B))

		norm = norm_L2(diff(yn0, yn1), A, B)
		if norm < eps:
			return N
		N += 1

if __name__ == "__main__":
	eps = 1e-5
	MAX_ITERATION_COUNT = 50
	stepSize = 0.005
	#	нижняя граница
	A = 1.0
	#	верхняя граница
	B = 2.0

	iterInd = 0
	N = getOptimalN(eps)
	print("N:", N)

	coefs = []
	while iterInd < MAX_ITERATION_COUNT:
		coefs = solveLSM(coefs, N, A, B)

		xn = getXn(coefs)
		norm = norm_L2(diff(xn, exect), A, B)
		if norm < eps:
			break

		iterInd += 1
		plotGraph(A, B, stepSize, coefs)

	print("Iter count:", iterInd)
	delta = norm_L2(diff(xn, exect), A, B)
	print("Delta:", delta)