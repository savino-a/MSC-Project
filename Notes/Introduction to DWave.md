# Introduction to Dwave
[Introduction to Dwave](https://docs.dwavesys.com/docs/latest/c_gs_1.html "Introduction to Dwave")

## Welcome to DWave

### D-Wave Software environment

Interface through web UI or through open-source tools that communicate with the Solver API (SAPI). The SAPI responsible for authentification, scheduling ...

#### Leap Quantum Cloud Service

It's the quantum cloud service from D-Wave : [Intro to Leap](https://docs.dwavesys.com/docs/latest/leap.html "Intro to Leap documentation")

#### Ocean SDK

To develop apps for quantum computers : [Intro to Ocean SDK](https://docs.ocean.dwavesys.com/en/stable/?_gl=1*1x4ae0q*_gcl_au*MTQ5NjE3MDI2Ni4xNzI5ODQ5MjU5*_ga*MTQwNzMyMzIyNC4xNzI5ODQ5MjU5*_ga_DXNKH9HE3W*MTcyOTg0OTI1OS4xLjEuMTcyOTg0OTk1OS42MC4wLjA.) / [Ocean SDK git](https://github.com/dwavesystems)

## Workflow: Formulation and Sampling

Main two steps of solving problems on quantum computers:
- Formulating problem as a cost function:
  - Functions that have lowest values for a good solution
- Finding good solutions to the problem by sampling
  - Samplers: processes that sample from low-energy states of objective functions. A variety of samplers available from D-Wave

### Objective/Cost functions

Mathematical expression of the **energy** of a system. Solution: global minimum of the function

#### Simple example

Solving: $x+1=2$
```math
E(x)=[2-(x+1)]^2 = (1-x)^2
```
Minimizing E is the same as getting close to equality therefore solving the equation.

## Models

To express a problem as an objective function that you can submit to a D-Wave sampler you typically use one of the models provided by Ocean SDK:
- Binary Quadratic Model: unconstraint and have binary variables. 
  - [Binary Quadratic Model](https://docs.ocean.dwavesys.com/en/stable/concepts/bqm.html?_gl=1*q63ya7*_gcl_au*MTQ5NjE3MDI2Ni4xNzI5ODQ5MjU5*_ga*MTQwNzMyMzIyNC4xNzI5ODQ5MjU5*_ga_DXNKH9HE3W*MTcyOTg0OTI1OS4xLjEuMTcyOTg1MDkzMy42MC4wLjA.#bqm-sdk)
- Constrained Quadratic Model: can be constrained and have binary, integer and real variables.
  -  [Constrained Quadratic Model](https://docs.ocean.dwavesys.com/en/stable/concepts/cqm.html?_gl=1*guerb4*_gcl_au*MTQ5NjE3MDI2Ni4xNzI5ODQ5MjU5*_ga*MTQwNzMyMzIyNC4xNzI5ODQ5MjU5*_ga_DXNKH9HE3W*MTcyOTg0OTI1OS4xLjEuMTcyOTg1MDkzMy42MC4wLjA.#cqm-sdk)
- Discret Quadratic Model: unconstraint and have discrete variables. Used when optimize over different options.
  -  [Discrete Quadradatic Model](https://docs.ocean.dwavesys.com/en/stable/concepts/dqm.html?_gl=1*1x82ace*_gcl_au*MTQ5NjE3MDI2Ni4xNzI5ODQ5MjU5*_ga*MTQwNzMyMzIyNC4xNzI5ODQ5MjU5*_ga_DXNKH9HE3W*MTcyOTg0OTI1OS4xLjEuMTcyOTg1MDkzMy42MC4wLjA.#dqm-sdk)
- Nonlinear Model: can be constrained and have binary and integer variables. For use with decision variables that represent a common logic. ie: travelling salesman.
  - [Nonlinear Model](https://docs.ocean.dwavesys.com/en/stable/concepts/nl_model.html?_gl=1*1l1clzb*_gcl_au*MTQ5NjE3MDI2Ni4xNzI5ODQ5MjU5*_ga*MTQwNzMyMzIyNC4xNzI5ODQ5MjU5*_ga_DXNKH9HE3W*MTcyOTg0OTI1OS4xLjEuMTcyOTg1MDkzMy42MC4wLjA.#nl-model-sdk)