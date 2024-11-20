# Modeling a train's acceleration
We are trying to determine the $\Delta v(v)$ of a train as approximating it as a constant is too big an approximation.

## Train dynamics:

**The fundamental law of train dynamics:**
```math
F_{wheel} = F_{resistance} + F_{weight} + F_{curves} + F_{acceleration}
```
```math
F_{wheel} = RAV + \frac{M_sg}{1000}i + \frac{M_sg}{1000} \frac{800}{\rho} + k M_s \gamma
```