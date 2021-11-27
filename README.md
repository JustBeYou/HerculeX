HerculeX
===

Hex game bot powered by reinforcement learning.

## OpenAI Gym Environment

The envrionment is based on [FirefoxMetzger's implementation](https://github.com/FirefoxMetzger/minihex) 
with a few tweaks:
 * separated classes into multiple files
 * implemented the random policy inside an agent class
 * improved the performance of the random agent by caching some arrays
 * customizable reward function

Observation space is described by: `(board, player) = (NxN matrix, {0, 1})`
Action space is described by: `(x, y)` where it is any position on the board.

**Note:** There is no support for the **swap rule** at the moment.
