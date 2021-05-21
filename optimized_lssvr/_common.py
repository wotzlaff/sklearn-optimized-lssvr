import numpy as np


def solve_rendered(qm, qv, rv, r0=0.0):
    qmiqv = np.linalg.solve(qm, qv)
    qmirv = np.linalg.solve(qm, rv)
    b = (qmiqv.dot(rv) - r0) / qmiqv.dot(qv)
    a = qmirv - b * qmiqv
    return a, b
