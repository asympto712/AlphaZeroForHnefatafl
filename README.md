# AlphaZeroForHnefatafl
Implementation of AlphaZero for Hnefatafl the board game (still in progress).

## The purpose of this project
We've all seen the daring, swashbuckling exploits of Google's DeepMind in the area of game-playing AI; In 2015, they swoop in with their AlphaGo. In 2016, their AlphaGo Zero, an even more general model applicable to Chess, Shogi, Go, defeated the human back-to-back Go champion. We still remember that nature issue came out with their big feat and how everybody was talking about it for weeks. Well, almost 10 years later, we were given the opportunity to study it and see if we can apply that to other games as a project in Scientific Computing class, held by Prof. Thorsten Koch from Zuse Institute Berlin. This game is different from games like chess in a way; it is a two player zero sum game, but two sides play according to different rules. The attacker tries to capture the king before he escapes to one of the four corners, and the defender attempts to safely escort the king to the exit utilizing his soldiers. 

  ### The Progress thus far
As of the time of writing this, 18th February 2025, we managed to run the whole thing, but haven't succeeded in building a model that can play on a decent level, much less playing competitively. However, the whole structure is there, so all we need to do is play around hyper parameters and slightly different implementations and see what approach works best.


## How to run: 
  1. make your python venv (I used python3.11) 
  2. activate the venv 
  3. pip install torch, numpy, maturin, etc.. 
  4. (on your terminal) maturin develop 
  5. should get a _azhnefatafl.~.pdy file under src/azhnefatafl/ No need to call maturin build

