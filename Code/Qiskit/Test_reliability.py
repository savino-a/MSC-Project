from QK_opt import *

Count = 0
for i in range(1, 101):
    pb = Qiskit_Problem(N=4, D=2, vmax=1, eff=False, p=1, dist_tolerance=0)
    pb._solve_(plot=False)
    if pb.resp_constraint:
        Count += 1
    print(i)
    print(Count / i)
print(Count)
