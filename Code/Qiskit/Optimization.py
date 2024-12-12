import qiskit
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
import qiskit_aer as aer


"""def constant_energy_model(
    N: int,  # Number of time steps
    distance_constraint: int,  # Total distance to be covered
    v_maximum: int,  # Maximum allowed velocity
    delta_V: int,  # Velocity increment per time step
    delta_t: int,  # Length of a single time step
) -> QuadraticProgram:

    # Create a new QuadraticProgram
    qp = QuadraticProgram()
    # Add binary variables to qp for each time step
    for i in range(N):
        qp.binary_var(name=f" x { i } ")
    # Define the objective function
    qp.minimize(linear=[delta_V**2] * N)
    # Constraint 1: Total distance must be covered
    qp.linear_constraint(
        linear=[(N - i) * delta_V * delta_t for i in range(N)],
        sense=" == ",
        rhs=distance_constraint,
        name=" distance_c ",
    )
    # V_max constraint
    qp.linear_constraint(
        linear=[delta_V] * N,
        sense=" <= ",
        rhs=v_maximum,
        name=" Speed_limit_constraint ",
    )
    return qp"""
