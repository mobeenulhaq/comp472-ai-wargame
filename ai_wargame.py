from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import Tuple, TypeVar, Type, Iterable, ClassVar
import random
import requests
import math

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000


class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4


class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker


class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3


##############################################################################################################

@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health: int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table: ClassVar[list[list[int]]] = [
        [3, 3, 3, 3, 1],  # AI
        [1, 1, 6, 1, 1],  # Tech
        [9, 6, 1, 6, 1],  # Virus
        [3, 3, 3, 3, 1],  # Program
        [1, 1, 1, 1, 1],  # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table: ClassVar[list[list[int]]] = [
        [0, 1, 1, 0, 0],  # AI
        [3, 0, 0, 3, 3],  # Tech
        [0, 0, 0, 0, 0],  # Virus
        [0, 0, 0, 0, 0],  # Program
        [0, 0, 0, 0, 0],  # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta: int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"

    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()

    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount


##############################################################################################################

@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row: int = 0
    col: int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
            coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
            coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string() + self.col_string()

    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()

    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row - dist, self.row + 1 + dist):
            for col in range(self.col - dist, self.col + 1 + dist):
                yield Coord(row, col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row - 1, self.col)
        yield Coord(self.row, self.col - 1)
        yield Coord(self.row + 1, self.col)
        yield Coord(self.row, self.col + 1)

    @classmethod
    def from_string(cls, s: str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if (len(s) == 2):
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None


##############################################################################################################

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src: Coord = field(default_factory=Coord)
    dst: Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string() + " " + self.dst.to_string()

    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row, self.dst.row + 1):
            for col in range(self.src.col, self.dst.col + 1):
                yield Coord(row, col)

    def is_up_or_left(self) -> bool:  # helper function for is_valid_move()
        """Checks whether dst coord is at the top and left of src coord"""
        return (self.dst.row < self.src.row and self.dst.col == self.src.col) or \
            (self.dst.row == self.src.row and self.dst.col < self.src.col)

    def are_diagonal(self) -> bool:  # helper function for is_valid_move()
        """Checks whether dst coord and src coord are diagonal"""
        if self.src.row != self.dst.row and self.src.col != self.dst.col:
            return True
        return False
    
    def are_adjacent(self) -> bool:  # helper function for is_valid_move()
        """Checks whether dst coord and src coord are adjacent"""
        for adj in self.src.iter_adjacent():
            if self.dst == adj:
                return True
        return False

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0, col0), Coord(row1, col1))

    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0, 0), Coord(dim - 1, dim - 1))

    @classmethod
    def from_string(cls, s: str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if (len(s) == 4):
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None


##############################################################################################################

@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth: int | None = 4
    min_depth: int | None = 2
    max_time: float | None = 5.0
    game_type: GameType = GameType.AttackerVsDefender
    alpha_beta: bool = True
    max_turns: int | None = 100
    randomize_moves: bool = True
    broker: str | None = None
    heuristic : str | None = 'e0'

    def get_filename(self):
        return f"gameTrace-{self.alpha_beta}-{str(int(self.max_time))}-{str(int(self.max_turns))}.txt"


##############################################################################################################

@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth: dict[int, int] = field(default_factory=dict)
    total_seconds: float = 0.0


##############################################################################################################

@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played: int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    nodes_visited: int = 0
    current_evaluations_per_depth: dict[int, int] = field(default_factory=dict)
    _attacker_has_ai: bool = True
    _defender_has_ai: bool = True

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim - 1
        self.set(Coord(0, 0), Unit(player=Player.Defender, type=UnitType.AI))
        self.set(Coord(1, 0), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(0, 1), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(2, 0), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(0, 2), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(1, 1), Unit(player=Player.Defender, type=UnitType.Program))
        self.set(Coord(md, md), Unit(player=Player.Attacker, type=UnitType.AI))
        self.set(Coord(md - 1, md), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md, md - 1), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md - 2, md), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md, md - 2), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md - 1, md - 1), Unit(player=Player.Attacker, type=UnitType.Firewall))

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord: Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord: Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord: Coord, unit: Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord, None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord: Coord, health_delta: int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def _engaged_in_combat(self, src_coord: Coord):  # helper function for is_valid_move()
        """Checks whether src unit is engaged in combat"""
        src_unit = self.get(src_coord)
        for adj in src_coord.iter_adjacent():
            adj_unit = self.get(adj)
            if adj_unit is not None:
                if src_unit.player != adj_unit.player:
                    return True
        return False

    def is_valid_move(self, coords: CoordPair) -> bool:
        """Validate a move expressed as a CoordPair."""
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False
        unit = self.get(coords.src)
        if unit is None or unit.player != self.next_player:
            return False
        # if both coords are equal then it's a valid self-destruction move
        if coords.src == coords.dst:
            return True
        if not coords.are_adjacent():
            return False
        if coords.are_diagonal():
            return False
        # Virus and Tech can move freely
        if unit.type.name in {"Virus", "Tech"}:
            return True
        # for AI, Firewall, Program
        vacant = self.get(coords.dst) is None  # destination is vacant
        eic = self._engaged_in_combat(coords.src)  # src unit engaged in combat
        up_or_left = coords.is_up_or_left()  # destination is up or left of source
        if unit.player == Player.Attacker:
            if vacant and (eic or not up_or_left):
                return False
        elif unit.player == Player.Defender:
            if vacant and (eic or up_or_left):
                return False
        return True

    def _self_destruct(self, src_coord: Coord):  # helper function for perform_move()
        """Executes the unit self-destruct action"""
        # modify health of all 8 surrounding units
        for coord in src_coord.iter_range(dist=1):
            if self.get(coord):
                self.mod_health(coord, -2)
        self.mod_health(src_coord, -9)

    def perform_move(self, coords: CoordPair) -> Tuple[bool, str]:
        """Validate and perform a move expressed as a CoordPair."""
        if self.is_valid_move(coords):
            s = self.get(coords.src)
            t = self.get(coords.dst)
            result = ""
            # case: action is self-destruct
            if s == t:
                self._self_destruct(coords.src)
                result += f"Self-destruct at {coords.src}\n{self}\n\n"
                return (True, result)
            # case: action is harmless movement
            elif t is None:
                self.set(coords.dst, s)
                self.set(coords.src, None)
                result += f"Move from {coords.src} to {coords.dst}\n{self}\n\n"
                return (True, result)
            # case: action is attack
            elif s.player != t.player:
                health_delta_t = -s.damage_amount(t)
                health_delta_s = -t.damage_amount(s)
                self.mod_health(coords.src, health_delta=health_delta_s)
                self.mod_health(coords.dst, health_delta=health_delta_t)
                result += f"Attack from {coords.src} to {coords.dst}\n{self}\n\n"
                return (True, result)
            # case: action is repair
            else:
                health_delta = s.repair_amount(t)
                if health_delta == 0:
                    # repair invalid
                    return (False, "invalid move")
                self.mod_health(coords.dst, health_delta=health_delta)
                result += f"Repair from {coords.src} to {coords.dst}\n{self}\n\n"
                return (True, result)
        return (False, "invalid move")

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')

    def human_turn(self):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success, result) = self.perform_move(mv)
                    print(f"Broker {self.next_player.name}: ", end='')
                    print(result)
                    if success:
                        write_game_state_to_file(filename=self.options.get_filename(), text=result)
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                (success, result) = self.perform_move(mv)
                if success:
                    print(f"Player {self.next_player.name}: ", end='')
                    print(result)
                    write_game_state_to_file(filename=self.options.get_filename(), text=result)
                    self.next_turn()
                    break
                else:
                    print("The move is not valid! Try again.")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.suggest_move()
        if mv is not None:
            (success, result) = self.perform_move(mv)
            if success:
                print(f"Computer {self.next_player.name}: ", end='')
                print(result)
                write_game_state_to_file(filename=self.options.get_filename(), text=result)
                self.next_turn()
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord, Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord, unit)

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        if self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker
        return Player.Defender

    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for (src, _) in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(move):
                    yield move.clone()
            move.dst = src
            yield move.clone()

    def get_unit_count(self, player: Player, unit_type: UnitType) -> int:  # for e0
        """Returns the count of a specific unit type for a given player."""
        count = 0
        for _, unit in self.player_units(player):
            if unit.type == unit_type:
                count += 1
        return count

    def get_aggregate_health(self, player: Player):  # for e1
        """Returns the difference in health between the attacker and the defender (total units)"""
        total_health = 0
        for _, unit in self.player_units(player):
            if unit.type == UnitType.AI:
                total_health += 10*unit.health
            else:
                total_health += unit.health
        return total_health

    def get_potential_damage_delta(self):  # for e1
        """Calculate the potential damage one type of adversary can inflict on the other"""
        attacker_potential_damage = 0
        for _, unit in self.player_units(Player.Attacker):
            for _, opp_unit in self.player_units(Player.Defender):
                attacker_potential_damage += Unit.damage_table[unit.type.value][opp_unit.type.value]

        defender_potential_damage = 0
        for _, unit in self.player_units(Player.Defender):
            for _, opp_unit in self.player_units(Player.Attacker):
                defender_potential_damage += Unit.damage_table[unit.type.value][opp_unit.type.value]

        return attacker_potential_damage - defender_potential_damage
    
    def get_distance_between_coord (self,Coord1 :Coord, Coord2 : Coord) :
        columne_distance = Coord1.col - Coord2.col
        row_distance = Coord1.row - Coord2.row
        return math.sqrt(columne_distance ** 2 + row_distance **2 )

    def get_distance_from_AI(self):
        # Calculate Distance of AI from Friendly Units
        attacker_ai_distance = 0
        defender_ai_distance = 0
        for coords,unit in self.player_units(Player.Attacker):
            if(unit.type == UnitType.AI):
                for friendly_coord,friendly_unit in self.player_units(Player.Attacker):
                    if(friendly_unit.type != UnitType.AI):
                        attacker_ai_distance += 1/(self.get_distance_between_coord(coords,friendly_coord))
                break
        
        for coords,unit in self.player_units(Player.Defender):
            if(unit.type == UnitType.AI):
                for friendly_coord,friendly_unit in self.player_units(Player.Defender):
                    if(friendly_unit.type != UnitType.AI):
                        defender_ai_distance += 1/(self.get_distance_between_coord(coords,friendly_coord))
                break
        return attacker_ai_distance - defender_ai_distance

    def evaluate(self, heuristic_type: str) -> int:
        """Evaluate the board state using the given heuristic. Currently only support e0, e1. TODO: add e2"""
        heuristic_value = 0
        if heuristic_type == 'e0':
            # For Attacker
            vp_attacker = self.get_unit_count(Player.Attacker, UnitType.Virus)
            tp_attacker = self.get_unit_count(Player.Attacker, UnitType.Tech)
            fp_attacker = self.get_unit_count(Player.Attacker, UnitType.Firewall)
            pp_attacker = self.get_unit_count(Player.Attacker, UnitType.Program)
            ai_attacker = self.get_unit_count(Player.Attacker, UnitType.AI)
            # For Defender
            vp_defender = self.get_unit_count(Player.Defender, UnitType.Virus)
            tp_defender = self.get_unit_count(Player.Defender, UnitType.Tech)
            fp_defender = self.get_unit_count(Player.Defender, UnitType.Firewall)
            pp_defender = self.get_unit_count(Player.Defender, UnitType.Program)
            ai_defender = self.get_unit_count(Player.Defender, UnitType.AI)

            heuristic_value = (
                    ((3 * vp_attacker) + (3 * tp_attacker) + (3 * fp_attacker) + (3 * pp_attacker) + (9999 * ai_attacker)) -
                    ((3 * vp_defender) + (3 * tp_defender) + (3 * fp_defender) + (3 * pp_defender) + (9999 * ai_defender))
            )
        elif heuristic_type == 'e1':
            aggregate_health_delta = self.get_aggregate_health(Player.Attacker) - self.get_aggregate_health(Player.Defender)
            potential_damage_delta = self.get_potential_damage_delta()
            heuristic_value = aggregate_health_delta + potential_damage_delta
        
        elif heuristic_type == 'e2':
            heuristic_value = self.get_distance_from_AI()
        
        return int(heuristic_value)

    def minimax(self, depth: int, alpha: float, beta: float, maximizing_player: bool, start_time: datetime, max_time: float) -> Tuple[int, CoordPair | None]:
        """minimax recursive algorithm with alpha-beta pruning."""

        current_elapsed_time = (datetime.now() - start_time).total_seconds()
        actual_depth = self.options.max_depth - depth
        if actual_depth not in self.stats.evaluations_per_depth:
            self.stats.evaluations_per_depth[actual_depth] = 0
        if actual_depth not in self.current_evaluations_per_depth:
            self.current_evaluations_per_depth[actual_depth] = 0

        self.nodes_visited += 1

        # If depth has been reached or the game is finished or max time has approached, return a heuristic value
        if self.is_finished():
            if self.has_winner() == Player.Attacker:
                self.stats.evaluations_per_depth[actual_depth] += 1
                self.current_evaluations_per_depth[actual_depth] += 1
                return (MAX_HEURISTIC_SCORE, None)
            elif self.has_winner() == Player.Defender:
                self.stats.evaluations_per_depth[actual_depth] += 1
                self.current_evaluations_per_depth[actual_depth] += 1
                return (MIN_HEURISTIC_SCORE, None)
        elif depth == 0 or current_elapsed_time > max_time:
            self.stats.evaluations_per_depth[actual_depth] += 1
            self.current_evaluations_per_depth[actual_depth] += 1
            return (self.evaluate(self.options.heuristic), None)

        best_move = None

        if maximizing_player:  # for the maximizing player
            value = float('-inf')
            for move in self.move_candidates():
                game_clone = self.clone()
                (success, result) = game_clone.perform_move(move)
                if success:
                    game_clone.next_turn()
                    eval_value, _ = game_clone.minimax(depth - 1, alpha, beta, False, start_time, max_time)
                    if eval_value > value:
                        value = eval_value
                        best_move = move

                    if value > beta:
                        break

                    alpha = max(alpha, value)
                else:
                    continue

            return value, best_move

        else:  # for the minimizing player
            value = float('inf')
            for move in self.move_candidates():
                game_clone = self.clone()
                (success, result) = game_clone.perform_move(move)
                if success:
                    game_clone.next_turn()
                    eval_value, _ = game_clone.minimax(depth - 1, alpha, beta, True, start_time, max_time)
                    if eval_value < value:
                        value = eval_value
                        best_move = move

                    if value < alpha:
                        break

                    beta = min(beta, value)
                else:
                    continue

            return value, best_move

    def random_move(self) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return (0, move_candidates[0], 1)
        else:
            return (0, None, 0)

    def suggest_move(self) -> CoordPair | None:
        """Suggest the next move using minimax alpha-beta."""

        max_depth = int(self.options.max_depth)
        max_time = self.options.max_time

        start_time = datetime.now()

        # Use the minimax algorithm to get the best move.
        score, move = self.minimax(max_depth, float('-inf'), float('inf'), self.next_player == Player.Attacker, start_time, max_time)  # Attacker will always be the initial maximizer

        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds

        text = ""

        print(f"Heuristic score: {score}")
        text += f"Heuristic score: {score}\n"
        cumulative_evals = 0
        for eval_count in self.stats.evaluations_per_depth.values():
            cumulative_evals += eval_count
        print(f"Cumulative evals: {cumulative_evals}")
        text += f"Cumulative evals: {cumulative_evals}\n"
        print(f"Cumulative Evals per depth: ", end='')
        text += "Cumulative Evals per depth: \n"
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            if k != 0:
                print(f"{k}:{self.stats.evaluations_per_depth[k]} ", end='')
                text += f"{k}:{self.stats.evaluations_per_depth[k]} "
        print()
        text += "\n"
        total_evals = sum(self.stats.evaluations_per_depth.values())
        cumulative_percent = 0
        print("Cumulative % Evals per depth: ", end='')
        text += "Cumulative % Evals per depth: \n"
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            if k != 0:
                percentage = (self.stats.evaluations_per_depth[k] / total_evals) * 100
                cumulative_percent += percentage
                print(f"{k}:{cumulative_percent:.2f} ", end='')
                text += f"{k}:{cumulative_percent:.2f} "
        print()
        text += "\n"
        if self.stats.total_seconds > 0:
            print(f"Eval perf.: {total_evals / self.stats.total_seconds / 1000:0.1f}k/s")
            text += f"Eval perf.: {total_evals / self.stats.total_seconds / 1000:0.1f}k/s\n"
        print(f"Average branching factor: {sum(self.current_evaluations_per_depth.values()) / self.nodes_visited}")
        text += f"Average branching factor: {sum(self.current_evaluations_per_depth.values()) / self.nodes_visited}\n"
        self.nodes_visited = 0
        self.current_evaluations_per_depth = {}
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        text += f"Elapsed time: {elapsed_seconds:0.1f}s\n"
        write_game_state_to_file(filename=self.options.get_filename(), text=text)
        return move

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played + 1:
                        move = CoordPair(
                            Coord(data['from']['row'], data['from']['col']),
                            Coord(data['to']['row'], data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None


##############################################################################################################

def write_game_state_to_file(filename, text):
    with open(filename, 'a') as file:
        file.write(text)


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--max_turns', type=float, help='maximum number of turns')
    parser.add_argument('--game_type', type=str, default="manual", help='game type: auto|attacker|defender|manual')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    parser.add_argument('--heuristic',type=str,default='e0')
    args = parser.parse_args()

    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp

    # set up game options
    options = Options(game_type=game_type)

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.max_turns is not None:
        options.max_turns = args.max_turns
    if args.broker is not None:
        options.broker = args.broker
    if args.heuristic is not None:
        options.heuristic = args.heuristic

    # create a new game
    game = Game(options=options)

    # write initial required game params to text file
    filename = options.get_filename()
    initial_file_text = f"timeout: {options.max_time}\nmax turns: {options.max_turns}\nplayer 1 = H & player 2 = H\n\n{game}\n"
    write_game_state_to_file(filename=filename, text=initial_file_text)

    # the main game loop
    while True:
        print()
        print(game)
        winner = game.has_winner()
        if winner is not None:
            print(f"{winner.name} wins!")
            write_game_state_to_file(filename=filename, text=f"{winner.name} wins in {game.turns_played} turns")
            break
        if game.options.game_type == GameType.AttackerVsDefender:
            game.human_turn()
        elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn()
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn()
        else:
            player = game.next_player
            move = game.computer_turn()
            if move is not None:
                game.post_move_to_broker(move)
            else:
                print("Computer doesn't know what to do!!!")
                exit(1)


##############################################################################################################

if __name__ == '__main__':
    main()
