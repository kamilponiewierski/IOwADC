import time
import random
from easyAI import TwoPlayerGame, AI_Player, Negamax # Import easyAI 
from negamax_no_pruning import NegamaxNoPruning  # Import AI
from expectiminimax import ExpectiMinimax # Import algorytmux Expectiminimax



class AIWithTimer(AI_Player):
    def __init__(self, algorithm):
        super().__init__(algorithm)
        self.total_decision_time = 0
        self.move_count = 0

    def ask_move(self, game):
        start = time.time()
        chosen_move = super().ask_move(game)
        end = time.time()

        self.total_decision_time += (end - start)
        self.move_count += 1

        return chosen_move

    def average_decision_time(self):
        return self.total_decision_time / self.move_count if self.move_count else 0


def create_stacks():
    total_pieces = 30
    min_stacks = 3
    min_pieces_per_stack = 4

    num_stacks = random.randint(min_stacks, total_pieces // min_pieces_per_stack)
    stacks = [min_pieces_per_stack] * num_stacks
    remaining_pieces = total_pieces - sum(stacks)

    while remaining_pieces > 0:
        index = random.randint(0, num_stacks - 1)
        stacks[index] += 1
        remaining_pieces -= 1

    return stacks


class NimGame(TwoPlayerGame):
    def __init__(self, players, starting_player, probabilistic, stacks=None):
        self.players = players
        self.stacks = stacks if stacks is not None else create_stacks()
        self.current_player = starting_player
        self.history = []
        self.probabilistic = probabilistic

    def possible_moves(self):
        return ["%d,%d" % (i + 1, j) for i in range(len(self.stacks)) for j in range(1, self.stacks[i] + 1)]

    def make_move(self, move):
        stack, amount = map(int, move.split(','))
        if self.probabilistic and random.random() < 0.1:
            amount = max(amount - 1, 1)
        self.history.append(amount)
        self.stacks[stack - 1] -= amount

    def unmake_move(self, move):
        stack, amount = map(int, move.split(','))
        self.stacks[stack - 1] += self.history.pop()

    def show(self):
        print("Aktualny stan stosów:", " ".join(map(str, self.stacks)))

    def win(self):
        return max(self.stacks) == 0

    def is_over(self):
        return self.win()

    def scoring(self):
        return 100 if self.win() else 0


if __name__ == "__main__":
    num_games = int(input("Podaj liczbę gier do rozegrania: "))
    depth1 = int(input("Podaj maksymalną głębokość dla Gracza 1: "))
    depth2 = int(input("Podaj maksymalną głębokość dla Gracza 2: "))
    algo1_type = input("Wybierz algorytm dla Gracza 1 (Negamax/BaseNegamax/ExpectiMiniMax): ").strip().lower()
    algo2_type = input("Wybierz algorytm dla Gracza 2 (Negamax/BaseNegamax/ExpectiMiniMax): ").strip().lower()
    probabilistic = input("Czy gra ma być probabilistyczna? (tak/nie): ").strip().lower() == "tak"
    
    match algo1_type:
        case "negamax":
            algo1 = Negamax
        case "basenegamax":
            algo1 = NegamaxNoPruning
        case "expectiminimax":
            algo1 = ExpectiMinimax
        case _:
            raise Exception("Niepoprawny algorytm")
    
    match algo2_type:
        case "negamax":
            algo2 = Negamax
        case "basenegamax":
            algo2 = NegamaxNoPruning
        case "expectiminimax":
            algo2 = ExpectiMinimax
        case _:
            raise Exception("Niepoprawny algorytm")

    initial_stacks = create_stacks()
    print(f"Początkowy układ stosów: {initial_stacks}")


    def play_simulation(algorithm1, algorithm2, depth1, depth2, num_games, probabilistic):
        ai1 = algorithm1(depth1)
        ai2 = algorithm2(depth2)
        player1 = AIWithTimer(ai1)
        player2 = AIWithTimer(ai2)
        wins_player1 = 0
        wins_player2 = 0

        for game_number in range(num_games):
            game = NimGame([player1, player2], 1 + game_number % 2, probabilistic, initial_stacks.copy())
            game.play(verbose=False)
            # print(f"Zwycięzcą jest Gracz {game.current_player}!")
            if game.current_player == 1:
                wins_player1 += 1
            else:
                wins_player2 += 1

        print(f"Gracz 1 wygrał: {wins_player1} razy")
        print(f"Gracz 2 wygrał: {wins_player2} razy")
        print(f"Średni czas decyzji Gracza 1: {player1.average_decision_time():.4f} sekundy")
        print(f"Średni czas decyzji Gracza 2: {player2.average_decision_time():.4f} sekundy")


    play_simulation(algo1, algo2, depth1, depth2, num_games, probabilistic)
