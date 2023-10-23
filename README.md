# AI Wargame

## Introduction

To run the game, execute the ai_wargame script using Python. The following command-line arguments can be passed:

--max_depth: Maximum search depth.

--max_time: Maximum search time. This is a float value that defines how long (in seconds) the game should spend searching for a move.

--max_turns: Maximum number of turns. This is a float value that limits the number of turns in a game.

--game_type: Type of game mode. Can be one of: auto, attacker, defender, or manual (default).


## Example command

```shell
python ai_wargame.py --max_time 10 --max_turns 25 --max_depth 3 --heuristic 2 --game_type auto
```


