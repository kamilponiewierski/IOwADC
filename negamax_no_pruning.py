"""
Implementation of Negamax algorithm without alpha-beta pruning.
"""

LOWERBOUND, EXACT, UPPERBOUND = -1, 0, 1
inf = float("infinity")


def negamax(game, depth, origDepth, scoring, tt=None):
    """
    Negamax algorithm implementation without alpha-beta pruning.
    """
    
    # Check transposition table
    lookup = None if (tt is None) else tt.lookup(game)
    if lookup is not None and lookup["depth"] >= depth:
        if depth == origDepth:
            game.ai_move = lookup["move"]
        return lookup["value"]

    # Base case: terminal node or depth limit reached
    if depth == 0 or game.is_over():
        return scoring(game) * (1 + 0.001 * depth)

    # Get possible moves and initialize best values
    possible_moves = game.possible_moves()
    best_value = -inf
    best_move = possible_moves[0]
    state = game

    # Search through all possible moves
    for move in possible_moves:
        # Make move and switch players
        game_copy = state.copy()
        game_copy.make_move(move)
        game_copy.switch_player()

        # Recursive negamax call
        current_value = -negamax(game_copy, depth-1, origDepth, scoring, tt)

        # Update best values
        if current_value > best_value:
            best_value = current_value
            best_move = move
            if depth == origDepth:
                state.ai_move = move

    # Store result in transposition table
    if tt is not None:
        tt.store(
            game=state,
            depth=depth,
            value=best_value,
            move=best_move,
            flag=EXACT
        )

    return best_value


class NegamaxNoPruning:
    """
    Game AI using Negamax algorithm without pruning.
    """
    
    def __init__(self, depth, scoring=None, tt=None):
        self.scoring = scoring
        self.depth = depth
        self.tt = tt

    def __call__(self, game):
        scoring = self.scoring if self.scoring else lambda g: g.scoring()
        negamax(game, self.depth, self.depth, scoring, self.tt)
        return game.ai_move