Look at other optimization algorithms

We want to look at a network. Train goes from A to B. How to profile that (minimum energy vs. shortest path) ?

Look at travelling salesman problem
- Looking at maximum speed on track, distance of track
- Take network and find shortest path

Then apply energy profiling (previous thesis) to find ac/dec profile 


Dynamic programming ?
- Is it overkill ?
- Recursion:
  - Next path I choose depends on previous paths and global cost
- Make sure what we are approximating is a good approximation for the problem. A good classifier
  - Difficult to find a good cost function


Start with binary optimisation, reproduce Cillian's results:
- Accelerating and coasting first
- Then regenerative braking, buildind on network etc.

**Prepare presentation/summary of the problem (QAOA github etc.)** \
Organize the meeting
Create an overleaf
Code in cplex