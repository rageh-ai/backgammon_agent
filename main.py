from backgammon import backgammon
import random
from mlp import ml_perceptron
import pickle



# Instantiate the backgammon object and the perceptron

agent = ml_perceptron(16, 1, 80)
i = 0
w = 0
b = 0

game_len = 0

while i < 10000:
    print("\nNEW GAME:")

    agent.v_t = 0.5
    agent.z_t_b1 = 0
    agent.z_t_b_2 = 0
    agent.z_t_w1=0

    agent.z_t_w2=0

    game = backgammon()
    # Iterate over the moves until a winner is determined
    while game.get_winner() is None:
        # Iterate over the moves directly from the moves attribute    
        curr_state = game.get_board()

        for move in game.moves:
            # Randomly generate a scalar value
            score = agent.passthrough(move)
            # Call the store_move function to store the move and scalar value
            game.score_move(move, float(score))

        # Check if a winner is determined after each iteration
        if game.get_winner() is not None:
           
            if game.get_winner() == "WHITE":
                agent.update(1,curr_state)
                print("WHITE WINS")
                w+=1
            

            else:
                agent.update(0, curr_state)
                print("BLACK WINS")
                b+=1
            break
        vt_1 = agent.passthrough(game.get_board())
        agent.update(vt_1, curr_state)
        game_len+=1

    i+= 1

with open("pickles/mlp_pickle.bin", "wb") as f:
    pickle.dump(agent, f)


print(f"Percetage of white wins {w/i:.2f}")
print(f"Average game length: {game_len/i: .2f} ")


