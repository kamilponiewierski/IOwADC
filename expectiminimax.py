import random

class ExpectiMinimax:
    def __init__(self, depth):
        self.depth = depth

    def search(self, game, depth, alpha, beta):
        """ Rekurencyjna funkcja wyszukująca najlepszy ruch dla gracza, przeciwnika lub obsługująca losowość. """
        
        # Jeśli osiągnięto maksymalną głębokość lub gra się zakończyła, zwracamy wartość oceny
        if depth == 0 or game.is_over():
            return game.scoring()
        
        # Węzeł MAX (gracz, który chce wygrać)
        if game.current_player == 1:
            best_value = -float('inf')
            for move in game.possible_moves():
                game.make_move(move)
                value = self.search(game, depth - 1, alpha, beta)
                game.unmake_move(move)
                
                best_value = max(best_value, value)
                alpha = max(alpha, best_value)

                if alpha >= beta:  # Odcięcie alfa-beta
                    break
            return best_value

        # Węzeł MIN (przeciwnik)
        elif game.current_player == 2:
            best_value = float('inf')
            for move in game.possible_moves():
                game.make_move(move)
                value = self.search(game, depth - 1, alpha, beta)
                game.unmake_move(move)
                
                best_value = min(best_value, value)
                beta = min(beta, best_value)

                if alpha >= beta:  # Odcięcie alfa-beta
                    break
            return best_value

        # Węzeł CHANCE (losowość)
        else:
            expected_value = 0
            probability = 0.9 if game.probabilistic else 1.0  # 90% szansy na normalny ruch
            
            for move in game.possible_moves():
                game.make_move(move)
                value = self.search(game, depth - 1, alpha, beta)
                game.unmake_move(move)
                expected_value += probability * value

            return expected_value / len(game.possible_moves())

    def __call__(self, game):
        """ Funkcja zwracająca najlepszy ruch do wykonania w danym stanie gry. """
        
        best_move = None
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        for move in game.possible_moves():
            game.make_move(move)
            value = self.search(game, self.depth - 1, alpha, beta)
            game.unmake_move(move)

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, best_value)

        return best_move
