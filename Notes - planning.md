# Problem  Solving
## Problem Solving vs Planning
Blindfold Hiker Test: We need feedback from the environment

## Planning and Executing
why? 

- Environment is stohastic
- Multiagent environments
- Partial Observability
- Unknown of hierarchical plan

Instead of planning on a space of world statesm, we plan in a space of belief spates

## Vacuum example
When everything is working fine, world is deterministic, easy to plan

When sensors break: Unobservable or centrlist world

How does the agent then represent the state of the world?

## Sensorless
we start with belief state. When we execure actions we can get knowledge of the environment even without sensing

**Conformant plans:** Pkan that reaches a goal without ever observing the world

## Partially Observable
When for example we have local sensing
- Can see what location it is in
- Can see what is going on in the current location
- It cannot see whether there is dirt in any other location

What does the act-observe cycle do to the belief states?
- For action, the size of the belief state will either stay the same or decrease. Number of states remains the same. 
- The observation works in the opposite way. The current belief state is partitioned into pieces. Observations cannot introduce a new state. Observation will not make us more confused than we were before. 

## Stochastic environment
e.g. Robot with slippery wheel
- Results of actions result in a state space larger than before, increasing uncertainty
- Observations bring the uncertainty down

## Infinite sequences
In a finite tree representation loop is formed to represent infinite sequences

## Finding a succesfull plan
Same sort of process with searching a tree

When unbounded solution: Every lead is Goal
Bounded Solution: No loops

## Mathematical notation

## Tracking the Predict Update Cycle


## Classical Planning
Representation language for dealing with states, actions and plans. Also an approach for dealing with the problem of complexity by factoring the world into variables

What states look  like
- **State space:** k boolean variables
- **World State**: Complete assignmemt to the variables
- **Belief State**: Complete Assignment, Partial Assignment, Arbitrary Formula in boolean logic


What actions looks like

Action schema: Action, Precondition, Effect

## Progression Search

## Regression search
We take the description of the goal state. What actions would lead to that state?

## Regressionvs Progression
Regression is much more efficient
Forward search is popular now as we can come up with good heuristics and can do heuristic search

## Plan Space search
Search through the state of plans, rather than search through the space of spates

We start from empty plan. Then we modify the plan 

## Sliding Puzzle Example
Finding heuristic: Throw out some prerequisites

## Situation Calculus
Goal: You can't express the notion of All in propositional languages like classical planning but you can in First Order Logic

Regular FOL with a set of conventions for how to represent states and actions

Actions: objects

Situations: onjects

Fluents are situations 
Not very popular anymore

Successor State Axioms
- Once we have described this in the ordinary language of first order logic
- We don't need any special programs to manipulate it and come up with the solution
- Because we already have theorem for first order logic

**Weakness**: We weren't able to distinguish between probable and improbable solutions


