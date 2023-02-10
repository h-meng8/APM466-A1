import numpy as np
import matplotlib.pyplot as plt
import numpy_financial as npf

result = np.loadtxt("bond.csv", delimiter=",", skiprows=1)
prices = result.T
N = prices.shape[0] # number of days
M = prices.shape[1] # number of bonds

ytm = np.zeros([N, M])
spot = np.zeros([N, M])

times = np.linspace(0.5, 5, num=10)
coupon = np.array([0.25, 0.75, 1.50, 1.25, 0.50, 0.25, 1.00, 1.25, 2.75, 2.00])
dates = [16, 17, 18, 19, 20, 23, 24, 25, 26, 27]

## YTM
for i in range(N):
    for j in range(M):
        c = coupon[j]/2
        p = prices[i, j]
        cf = np.zeros(j + 2)
        cf[0] = -p
        cf[-1] = c + 100
        for k in range(1, j+1):
            cf[k] = c
        #print(cf)
        y = npf.irr(cf)
        ytm[i, j] = 2*y

for i in range(N):
    date = 'Jan ' + str(dates[i])
    plt.plot(times, ytm[i, :], label=date)
plt.legend(loc="upper right")
plt.xlabel("Time to Maturity")
plt.ylabel("Yield")
plt.title("YTM Curve")
plt.show()


## Spot curve

for i in range(N):
    for j in range(M):
        sum = 0
        for k in range(j):
            sum += coupon[j]/2 * np.exp(-spot[i, k] * times[k])
        r = -np.log((prices[i, j] - sum)/(100 + coupon[j]/2))/times[j]
        spot[i, j] = r
#print(spot)

for i in range(N):
    date = 'Jan ' + str(dates[i])
    plt.plot(times, spot[i, :], label=date)
plt.legend(loc="upper right")
plt.xlabel("Time to Maturity")
plt.ylabel("Spot Rate")
plt.title("Spot Curve")
plt.show()

## forward rate
forward = np.zeros([N, 4])
x_range = np.array([1, 2, 3, 4])
for i in range(N):
    for j in range(1, 5):
        f = -(-spot[i, 2*j+1]*(times[2*j+1]-1)+spot[i, 2*j]*(times[2*j]-1))/ 0.5
        forward[i, j-1] = f
for i in range(N):
    date = 'Jan ' + str(dates[i])
    plt.plot(x_range, forward[i, :], label=date)
plt.legend(loc="upper right")
plt.xlabel("Time to Maturity")
plt.ylabel("Forward Rate")
plt.title("Forward Curve")
plt.show()

## 5) Covariance matrix

mat_ytm = ytm[:, [1, 3, 5, 7, 9]].T ## row is variable
mat_forward = forward.T

x_ytm = np.zeros([5, N-1])
x_forward = np.zeros([4, N-1])

for i in range(5):
    for j in range(N-1):
        x_ytm[i, j] = np.log(mat_ytm[i, j+1]/mat_ytm[i, j])

for i in range(4):
    for j in range(N-1):
        x_forward[i, j] = np.log(mat_forward[i, j+1]/mat_forward[i, j])


cov_ytm = np.cov(x_ytm)
cov_forward = np.cov(x_forward)

print(cov_ytm)
print(cov_forward)

## 6) Eigenvalues and Eigenvectors

ytm_evalue, ytm_evec = np.linalg.eig(cov_ytm)
forward_evalue, forward_evec = np.linalg.eig(cov_forward)

print("YTM Eigenvalues:", ytm_evalue)
print("YTM Eigenvectors:", ytm_evec)
print("Forward Eigenvalues:", forward_evalue)
print("Forward Eigenvectors:", forward_evec)
