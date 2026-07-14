# Minisim PyTrainer


This is a simulation, RL, and education project that I developed from scratch to learn the fundamentals.
It's a good exercise I recommend for everyone. Your basic imports are the Python time and math libraries. 
Codex is helpful for your web parts, but you won't learn unless you write the simulator, environment, reward function, and q-learning scripts
from scratch. 

## Details
It's like a pivot table.
Uses Q-learning with Q-table. No models, just exploration and inferred rewards. Every position the world and agent can be in is an index in the Q-Table. 
That's what a row in your table is. Imagine a lame hopskotch that's just a 1 x 4 grid, so 4 positions. You can go forward or backward.

So your Q-Table would be shaped like this:
~~~
[ position ] [ forward ] [  backward  ]
[ square 1 ] [ number  ] [   number   ]
[ square 2 ] [ number  ] [   number   ]
[ square 3 ] [ number  ] [   number   ]
[ square 4 ] [ number  ] [   number   ]
~~~

As you can see, what it does is map a state to the actions and an updatable value for that state.
The numbers would start out as all zero. Your agent is just a script to advance the simulation with an action and get a state back.

You could design the flow like this:
~~~
curr_state = square 1
new_state, reward = sim.take_action(forward)
table[curr_state][forward] = reward
curr_state = new_state
~~~

**That's it, you've completed one iteration of a learning problem explored in simulation! Congratulations!**

## Getting Started


1. Make a file called simulator.py and describe the velocity of a ball falling. I recommend using a generator so you can call next on it for a single timestep.
The timestep interval gets the new position as described by your velocity equation
2. Store these positions in an array. Plot the trajectory with matplotlib to validate it works.
3 Add collision and a rectangle, so that when the ball's position enters the rectangular, the velocity flips in the opposite direction.

It has to be fast. Write your code with consideration. From there, you'll have enough intuition to build the next interface: environment.py, 
which exposes a discrete state and action interface.

## /pytrainer

This is the main repo. Go there. The other folders are an integration with a real robot to be continued at another time. 
Spoiler: Don't add a jump action if you want to use your policy in any physical setting.

