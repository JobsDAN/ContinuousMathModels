import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt

## Задача здесь уже сформулирована как (4.18)-(4.19), метод наименьших квадратов


#точное значение функции в t
def exect(t):
	return t**5

### тригонометрические полиномы
def e(k, t):
	return np.sin(np.pi * k * (t - 1))

def e1(k, t):
	return np.pi * k * np.cos(np.pi * k * (t - 1))
# вторая производная
def e2(k, t):
	return -(np.pi**2 * k**2 * np.sin(np.pi * k * (t - 1)))

## функция замены
def G(t):
	return 31 * t - 30

## начальное приближение
def getY0(t):
	return 0.0
def getY0Diff1(t):
	return 0.0
def getY0Diff2(t):
	return 0.0

## значение функции на прошлой итерации
def getYn(t, coefs):
	if len(coefs) == 0:
		return getY0(t)

	accum = 0.0
	for k in range(len(coefs)):
		accum += coefs[k] * e(k + 1, t)
	return accum

## оператор А
def operator(k, t):
	return -e2(k, t) + 6 * t**(-10) * (getY0(t) + G(t))**2 * e(k, t)

## интеграл, стоящий слева от знака равенства
def integrate_left(k, j, A, B):
	return sp.quad(lambda t: operator(k, t) * operator(j, t), A, B)[0]

## интеграл, стоящий справа от знака равенства
def integrate_right(j, prevCoefsAk, A, B):
	return sp.quad(lambda t: (6 * t**(-10) * (getY0(t) + G(t))**2 * getYn(t, prevCoefsAk) - 2 * t**(-10) * (getY0(t) + G(t))**3 - 10 * t**2) *
						     operator(j, t),
							 A, B)[0]

def plotGraph(A, B, stepSize, coefsAk):
	stepCount = int((B - A) / stepSize + 1)
	x_points = np.arange(A, B + stepSize, stepSize)

	exectX = []
	approximateX = []
	for i in range(stepCount):
		t = A + i * stepSize
		exectX.append(exect(t))

		#Xn(t) = Yn(t) + G(t)
		xn = getYn(t, coefsAk) + G(t)
		approximateX.append(xn)

	plt.plot(x_points, exectX, 'b', x_points, approximateX, 'r')
	plt.show()

def solveLSM(prevCoefs, iterCount, A, B):
	# матрица коэфициентов
	coefs = []
	# вектор правой части СЛАУ
	f = []

	# пишу на питоне, а получается С#
	for j in range(1, iterCount + 1):
		equation = []
		# коэффициенты для каждого уравнения
		for k in range(1, iterCount + 1):
			equation.append(integrate_left(k, j, A, B))


		coefs.append(equation)
		f.append(integrate_right(j, prevCoefs, A, B))

	return np.linalg.solve(coefs, f)

def getMaxYn(t, coefs):
	accum = 0.0
	for k in range(len(coefs)):
		accum += coefs[k] * e(k + 1, t)
	return accum


def getMaxYnDiff1(t, coefs):
	accum = 0.0
	for k in range(len(coefs)):
		accum += coefs[k] * e1(k + 1, t)
	return accum

def getMaxYnDiff2(t, coefs):
	accum = 0.0
	for k in range(len(coefs)):
		accum += coefs[k] * e2(k + 1, t)
	return accum


def norm(coefs):
	diff0 = opt.fminbound(lambda x, coefs: -getMaxYn(x, coefs), 1, 2, args=(coefs, ))
	diff1 = opt.fminbound(lambda x, coefs: -getMaxYnDiff1(x, coefs), 1, 2, args=(coefs, ))
	diff2 = opt.fminbound(lambda x, coefs: -getMaxYnDiff2(x, coefs), 1, 2, args=(coefs, ))
	return abs(diff0) + abs(diff1) + abs(diff2)

def getOptimalN(eps):
	return 5
	#не работает
	N = 1
	while True:
		coefs_l = solveLSM([], N, A, B)
		normN = norm(coefs_l)
		coefs_l = solveLSM([], N + 1, A, B)
		normNPlusOne = norm(coefs_l)

		print(abs(normN - normNPlusOne))
		if abs(normN - normNPlusOne) < eps:
			return N
		N += 1

eps = 0.001
MAX_ITERATION_COUNT = 50
stepSize = 0.005
#	нижняя граница
A = 1.0
#	верхняя граница
B = 2.0

iterInd = 0
shouldContinue = True
N = getOptimalN(eps)

coefs = []
while shouldContinue and iterInd < MAX_ITERATION_COUNT:
	coefs = solveLSM(coefs, N, A, B)

	# ниже должна быть логика, которая отвечает за прекращение работы алгоритма,
	# в случае, если найдено решение, удовлетворяющее заданному Eps
	# ???. мера пространства С^2 = max(|xn - ex|) + max(|(xn - ex)'|) + max(|(xn - ex)''|)
	#
	# на данный момент, это полная хуита(инфа 47%)
	#shouldContinue = False
	#for i in range(STEP_COUNT):
	#	t = A + i * STEP_SIZE
	#	xn = getYn(t) + G(t)
	#	if abs(xn - exect(t)) > Eps:
	#		shouldContinue = True
	#		break

	iterInd += 1
	print(iterInd)
	plotGraph(A, B, stepSize, coefs)
