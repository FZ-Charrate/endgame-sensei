#This is the engine that actually runs the chess game, we'll be using the python chess library.
#input : RL agent's moves, output : the new state of the game after the move is made.
import chess
import random
import chess.engine

class ChessEngine:

    # Standard piece values used for reward shaping (capture value, material lost, etc.)
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,  # king is never "captured", so its value doesn't matter here
    }

    def __init__(self, fen=None):
        self.board = chess.Board(fen) if fen else chess.Board()

    def legal_moves(self):
        return list(self.board.legal_moves)

    def push(self, move):
        self.board.push(move)

    def is_game_over(self):
        return self.board.is_game_over()

    def get_board(self):
        return self.board

    def get_fen(self):
        return self.board.fen()

    def get_state(self):
        return {
            "board": self.board,
            "fen": self.board.fen(),
            "legal_moves": list(self.board.legal_moves),
            "is_game_over": self.board.is_game_over(),
            "result": self.board.result() if self.board.is_game_over() else None
        }

    # ---- Reward-related helpers (Option B: outcome-based, not lookahead) ----
    # option A was to calculate rewards based on what could possibly happen next, but that's too complicated and will stall the learning process, because we want to give our agent rules but not too many rules in such way that the learning becomes generic.
    # option B is to calculate rewards based on the outcome of the move (the next state of the board after th move and the opponent's reply), which is simpler and more effective for our purposes.

    def get_captured_value(self, board_before, board_after, captured_color):
        """
        Returns how much material `captured_color` lost between board_before
        and board_after, by comparing total piece value before vs after.
        """
        value_before = self._total_material_value(board_before, captured_color)
        value_after = self._total_material_value(board_after, captured_color)
        return value_before - value_after
 
    def _total_material_value(self, board, color):
        """
        Sums PIECE_VALUES for every piece of `color` currently on `board`.
        """
        total = 0
        for piece in board.piece_map().values():
            if piece.color == color:
                total += self.PIECE_VALUES[piece.piece_type]
        return total
 
    def gave_check(self, board_after_agent_move):
        """
        Returns True if the opponent is in check right after the agent's move
        (i.e. we just checked them - good for us).
        """
        return board_after_agent_move.is_check()
 
    def received_check(self, board_after_opponent_move):
        """
        Returns True if the agent is in check right after the opponent's reply
        (i.e. they just checked us - bad for us).
        """
        return board_after_opponent_move.is_check()
 
    def get_mobility_score(self, board_before_agent_move, board_after_agent_move, agent_color):
        """
        Small positional nudge: rewards moves that increase the agent's own
        future options (legal move count) rather than cramping its own pieces.
        Stays small relative to captures/checks via the MOBILITY_WEIGHT scaling.
        """
        before_count = self._count_legal_moves_for(board_before_agent_move, agent_color)
        after_count = self._count_legal_moves_for(board_after_agent_move, agent_color)
 
        MOBILITY_WEIGHT = 0.02
        return (after_count - before_count) * MOBILITY_WEIGHT #Small reward to incourage mobility, but previlige captures and checks, so the agent seeks more mobility, when it has nothing meaningful to do (no captures or checks available).
 
    def _count_legal_moves_for(self, board, color):
        """
        Counts legal moves for `color` on `board`, regardless of whose turn
        it actually is on that board.
        """
        board_copy = board.copy() #another instance of the board, so we can change the turn without affecting the original board.
        board_copy.turn = color
        return len(list(board_copy.legal_moves))
 
    def get_game_outcome_reward(self, board_after_agent_move, board_after_opponent_move, agent_color):
        """
        Returns a large reward/penalty if the game is over.

        If board_after_opponent_move is None, the game already ended right
        after the agent's own move (e.g. the agent delivered checkmate, or
        stalemated the position) - evaluate board_after_agent_move instead.
        """
        final_board = board_after_opponent_move if board_after_opponent_move is not None else board_after_agent_move

        if not final_board.is_game_over():
            return 0

        if final_board.is_checkmate():
            # side to move is the one who got mated
            mated_color = final_board.turn
            return 100 if mated_color != agent_color else -100

        # stalemate, insufficient material, repetition, etc.
        return -5

    def get_reward(self, board_before_agent_move, board_after_agent_move, board_after_opponent_move, agent_color, opponent_color):
        captured = self.get_captured_value(board_before_agent_move, board_after_agent_move, opponent_color)
        checked_them = self.gave_check(board_after_agent_move)
        mobility = self.get_mobility_score(board_before_agent_move, board_after_agent_move, agent_color)
        outcome = self.get_game_outcome_reward(board_after_agent_move, board_after_opponent_move, agent_color)

        reward = 0.0
        reward += captured
        reward += 1 if checked_them else 0
        reward += mobility
        reward += outcome

        if board_after_opponent_move is not None:
            lost = self.get_captured_value(board_after_agent_move, board_after_opponent_move, agent_color)
            got_checked = self.received_check(board_after_opponent_move)
            reward -= lost
            reward -= 1 if got_checked else 0

        return reward
    

#------------ Define some policies for the engine -----------------



def random_policy(board):
    """
    Picks a uniformly random legal move. Used as the baseline opponent.
    """
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves)


def stockfish_policy(board, engine_path, time_limit=0.1):
    """
    Asks a real Stockfish engine for its best move.

    engine_path: path to the stockfish executable on your machine.
    time_limit: seconds Stockfish is allowed to think (small for fast training games).

    TODO: this opens a new engine process every call, which is slow if used
    for many moves in a row. Once this works correctly, consider passing in
    an already-open `engine` object instead of a path, so it's opened once
    outside the game loop and reused across moves.
    """
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        result = engine.play(board, chess.engine.Limit(time=time_limit))
        return result.move


def rl_policy(board, agent, epsilon=0.1):
    """
    Epsilon-greedy move selection using the RL agent.

    With probability `epsilon`, picks a random legal move (exploration).
    Otherwise, asks the agent to score the available legal moves and picks
    the one it currently rates highest (exploitation).

    `agent` is expected to expose something like agent.get_q_value(board, move)
    returning a single number (higher = better) for a given board/move pair.
    Adjust this once the actual agent class interface is finalized.
    """
    legal_moves = list(board.legal_moves)

    if random.random() < epsilon:
        return random.choice(legal_moves)

    best_move = None
    best_value = float("-inf")
    for move in legal_moves:
        value = agent.get_q_value(board, move) #TODO: This should be defined later in the agent class
        if value > best_value:
            best_value = value
            best_move = move

    return best_move