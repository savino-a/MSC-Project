# Formulation of Alstom Problem

## Variables:
We have $N$ decision steps, therefore, we have $\forall i \in [0,N-1]$
- Speeds: $(v_i)_i \in [0;v_{max}]$
- Decisions: $(D_i)_i \in [0;2]$ 
- Useful variables: $(\delta_i)_i \in \mathcal{B}$ 


## Cost function:
The cost or energy used is: $C = \sum_{i=0}^{N-1} C_i$,  where:
$$C_i = f(v_i,D_i) = \frac{P_{max}}{C_{max}} \Delta t \ (1-exp(\frac{-v_i}{v_{max} + 0.001})) \ (tanh(100 \ D_i)) \ (1-exp(\frac{-D_i}{Dec_{max} + 0.00001}))   $$

### Transformed function:

We simplify our problem by approximating the $C_i$ by using a polynomial:
$$\forall i : \ C_i(v_i,D_i) = C_0 + C_1 v_i + C_2 (D_i-1) + C_3 \ v_i \times (D_i -1)  $$

## Constraints:

### Speed constrained by acceleration:

- $v_0=0$
- $v_{N-1}=0$
- $\forall i \in [1,N-2] : v_{i+1} = v{i} + \gamma_i \ \Delta t $, where $\gamma_i = \gamma (v_i,D_i)$ is the acceleration ($km/h/s$)

#### Effort functions

We have $F_{\text{traction}}$ and $F_{\text{braking}}$ the acceleration and braking forces respectively, where:

- $F_{\text{traction}} (v_i) = \begin{cases} -354.1 \ v_i + 2.44 \times 10^5 \ \ \ \ \ \text{     if  } v_i \in [0;50] \ kph \\ -881.3 \ v_i + 2.704 \times 10^5 \ \ \ \ \  \text{  if  } v_i \in [50;60] \ kph \\ -0.05265 \ v_i^3 + 28.78 \ v_i^2 - 5603 \ v_i + 4.566 \times 10^ 5 \ \ \ \ \  \text{  if  } v_i \in [60;220] \ kph \\ \end{cases}$

- $F_{\text{braking}} (v_i) = \begin{cases} 9925 \ v_i + 1.243 \ \ \ \ \ \text{     if  } v_i \in [0;20] \ kph \\ 2.039 \times 10^{-13} \ v_i + 1.985 \times 10^5 \ \ \ \ \  \text{  if  } v_i \in [20;100] \ kph \\ 5.389 \ v_i^2 - 2583 \ v_i + 4.012 \times 10^ 5 \ \ \ \ \  \text{  if  } v_i \in [60;220] \ kph \\ \end{cases}$

##### Approximating the F

We approximate the effort functions as polynomials:

- $F_{\text{traction}}(v_i) = A_1 + A_2 v_i$
- $F_{\text{braking}}(v_i) = B_1 + B_2 v_i$

#### Determining the acceleration ($\gamma_i$)

##### Fundamental principle of train dynamics

The fundamental principle of train dynamics gives us:
$$ F_i = RAV_i + \frac{M_s g}{1000} (\alpha + \frac{800}{\rho}) + kM_s \gamma_i $$
Where:
- $\gamma_i$ the residual acceleration
- $k$ the coefficient to consider the rotating mass of the train
- $\alpha$ the angle of the track
- $\frac{800}{\rho}$ the term corresponding to the curvature of the track
- $RAV_i$ the resistance to advancement of the train

We consider the track is perfectly straight and horizontal, therefore:
$$ F(v_i) = RAV(v_i) + kM_s \gamma_i $$

We also have the Davies formula:
$$ RAV(v_i) = A + B v_i + C v_i^2 $$
Where $A,B,C$ constants

##### The $\delta_i$ variable
We use the $\delta_i$ variable to determine whether are accelerating or braking, $\delta_i = \begin{cases} 1 \ if \ D_i \ge 1 \\ 0 \ otherwise\end{cases} $

Therefore, with this, the force applied at the wheels at time step $i$ is:
$$F(v_i,D_i) = (D_i-1) (\delta_i \ F_{\text{traction}}(v_i) + (\delta_i-1)F_{\text{braking}}(v_i))$$

##### Final constraint

We therefore have:
$$ F_i = F(v_i,D_i) = RAV_i + kM_s\gamma_i = (D_i-1) (\delta_i \ F_{\text{traction}}(v_i) + (\delta_i-1)F_{\text{braking}}(v_i)) $$

This gives us the acceleration at time step i:
$$\gamma_i = \gamma(v_i,D_i) =\frac{(D_i-1) (\delta_i \ F_{\text{traction}}(v_i) + (\delta_i-1)F_{\text{braking}}(v_i)) - RAV_i}{k M_s} $$

Also written as:

$$\gamma_i = \frac{F_i - RAV_i}{kM_s}$$

### Distance constraint:

We have $D$ the distance to travel, therefore, with $D_j = \Delta t \sum_{i=0}^j v_i  $, this gives us:
$$\Delta t \sum_{i=0}^{N-1} v_i \approx D$$

