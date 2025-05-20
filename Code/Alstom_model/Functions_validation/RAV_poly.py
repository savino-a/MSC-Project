A, B, C, m, v_max, p_max, c_max = 0.632, 40.7, 3900, 320000, 220, 4305220, 320000


def P(v):
    return A + B * v + C * v**2
