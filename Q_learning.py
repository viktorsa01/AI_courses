import numpy as np
import matplotlib.pyplot as plt


rewards: tuple[float, ...] = (-0.1, -0.1, -0.1, -0.1,
                              -0.1, -1.0, -0.1, -1.0,
                              -0.1, -0.1, -0.1, -1.0,
                              -1.0, -0.1, -0.1, 1.0)

transition_matrix: tuple[tuple[tuple[tuple[float, int]]]] = \
    ((((.9, 0), (.1, 4)), ((.1, 0), (.8, 4), (.1, 1)),
      ((.1, 4), (.8, 1), (.1, 0)), ((.1, 1), (.9, 0))),
     (((.1, 1), (.8, 0), (.1, 5)), ((.1, 0), (.8, 5), (.1, 2)),
      ((.1, 5), (.8, 2), (.1, 1)), ((.1, 2), (.8, 1), (.1, 0))),
     (((.1, 2), (.8, 1), (.1, 6)), ((.1, 1), (.8, 6), (.1, 3)),
      ((.1, 6), (.8, 3), (.1, 2)), ((.1, 3), (.8, 2), (.1, 1))),
     (((.1, 3), (.8, 2), (.1, 7)), ((.1, 2), (.8, 7), (.1, 3)),
      ((.1, 7), (.9, 3)), ((.9, 3), (.1, 2))),
     (((.1, 0), (.8, 4), (.1, 8)), ((.1, 4), (.8, 8), (.1, 5)),
      ((.1, 8), (.8, 5), (.1, 0)), ((.1, 5), (.8, 0), (.1, 4))),
     (((1.0, 5),), ((1.0, 5),), ((1.0, 5),), ((1.0, 5),)),
     (((.1, 2), (.8, 5), (.1, 10)), ((.1, 5), (.8, 10), (.1, 7)),
      ((.1, 10), (.8, 7), (.1, 2)), ((.1, 7), (.8, 2), (.1, 5))),
     (((1.0, 7),), ((1.0, 7),), ((1.0, 7),), ((1.0, 7),)),
     (((.1, 4), (.8, 8), (.1, 12)), ((.1, 8), (.8, 12), (.1, 9)),
      ((.1, 12), (.8, 9), (.1, 4)), ((.1, 9), (.8, 4), (.1, 8))),
     (((.1, 5), (.8, 8), (.1, 13)), ((.1, 8), (.8, 13), (.1, 10)),
      ((.1, 13), (.8, 10), (.1, 5)), ((.1, 10), (.8, 5), (.1, 8))),
     (((.1, 6), (.8, 9), (.1, 14)), ((.1, 9), (.8, 14), (.1, 11)),
      ((.1, 14), (.8, 11), (.1, 6)), ((.1, 11), (.8, 6), (.1, 9))),
     (((1.0, 11),), ((1.0, 11),), ((1.0, 11),), ((1.0, 11),)),
     (((1.0, 12),), ((1.0, 12),), ((1.0, 12),), ((1.0, 12),)),
     (((.1, 9), (.8, 12), (.1, 13)), ((.1, 12), (.8, 13), (.1, 14)),
      ((.1, 13), (.8, 14), (.1, 9)), ((.1, 14), (.8, 9), (.1, 12))),
     (((.1, 10), (.8, 13), (.1, 14)), ((.1, 13), (.8, 14), (.1, 15)),
      ((.1, 14), (.8, 15), (.1, 10)), ((.1, 15), (.8, 10), (.1, 13))),
     (((1.0, 15),), ((1.0, 15),), ((1.0, 15),), ((1.0, 15),)))


def valid_state(state: int) -> bool:
    return isinstance(state, (int, np.signedinteger)) and 0 <= state < 16


def valid_action(action: int) -> bool:
    return isinstance(action, (int, np.signedinteger)) and 0 <= action < 4

# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------- Nothing you need to do or use above this line. -----------------------------------------
# --------------------------------------------------------------------------------------------------------------------------


# Use these constants when you implement the value iteration algorithm.
# Do not change these values, except DETERMINISTIC when debugging.
N_STATES: int = 16
N_ACTIONS: int = 4
EPSILON: float = 1e-8
GAMMA: float = 0.9
DETERMINISTIC: bool = False


def get_next_states(state: int, action: int) -> list[int]:
    """
    Fetches the possible next states given the state and action pair.
    :param state: a number between 0 - 15.
    :param action: an integer between 0 - 3.
    :return: A list of possible next states. Each next state is a number between 0 - 15.
    """
    assert valid_state(state), \
        f"State {state} must be an integer between 0 - 15."
    assert valid_action(action), \
        f"Action {action} must be an integer between 0 - 3."
    next_state_probs = {next_state: trans_prob for trans_prob,
                        next_state in transition_matrix[state][action]}
    if DETERMINISTIC:
        return [max(next_state_probs, key=next_state_probs.get)]
    return next_state_probs.keys()


def get_trans_prob(state: int, action: int, next_state: int) -> float:
    """
    Fetches the transition probability for the next state
    given the state and action pair.
    :param state: an integer between 0 - 15.
    :param action: an integer between 0 - 3.
    :param outcome_state: an integer between 0 - 15.
    :return: the transition probability.
    """
    assert valid_state(state), \
        f"State {state} must be an integer between 0 - 15."
    assert valid_action(action), \
        f"Action {action} must be an integer between 0 - 3."
    assert valid_state(next_state), \
        f"Next state {next_state} must be an integer between 0 - 15."
    next_state_probs = {next_state: trans_prob for trans_prob,
                        next_state in transition_matrix[state][action]}
    # If the provided next_state is invalid.
    if next_state not in next_state_probs.keys():
        return 0.
    if DETERMINISTIC:
        return float(next_state == max(next_state_probs, key=next_state_probs.get))
    return next_state_probs[next_state]


def get_reward(state: int) -> float:
    """
    Fetches the reward given the state. This reward function depends only on the current state.
    In general, the reward function can also depend on the action and the next state.
    :param state: an integer between 0 - 15.
    :return: the reward.
    """
    assert valid_state(state), \
        f"State {state} must be an integer between 0 - 15."
    return rewards[state]


def get_action_as_str(action: int) -> str:
    """
    Fetches the string representation of an action.
    :param action: an integer between 0 - 3.
    :return: the action as a string.
    """
    assert valid_action(action), \
        f"Action {action} must be an integer between 0 - 3."
    return ("left", "down", "right", "up")[action]


if __name__ == '__main__':
    # TODO: Your code goes here. You can define additional
    # functions and variables anywhere in the file as you see fit.
    
    def Q(s,a):
        q = 0 
        for next_state in get_next_states(s, a):
            q += get_trans_prob(s, a, next_state)*(get_reward(next_state) + GAMMA*U[next_state])
        
        return q
    
    def get_Neighbours(state):
        neigh = []
        if state - 4 >= 0 :
            neigh.append(state-4)
        if state + 4 < 16 :
            neigh.append(state+4)
        if state % 4 != 0:
            neigh.append(state-1)
        if state %4 != 3:
            neigh.append(state+1)
        return neigh
        
    
    #initialize array of utilities
    U = np.zeros(N_STATES)
    U_next = np.zeros(N_STATES)
    
    for i in range(10000):      
        U = np.copy(U_next)
        delta = 0
        for state in range(N_STATES):
            
            maxQ = -100
            for action in range(N_ACTIONS):
                q = Q(state, action)
                if q > maxQ:
                    maxQ = q
            U_next[state] = maxQ
            
            if abs(U_next[state] - U[state]) > delta:
                delta = abs(U_next[state] - U[state])
                
        if delta <= EPSILON*(1-GAMMA)/GAMMA:
            print(f"The algorithm used {i} iterations")
            break
    
    path = list({0})
    for i in range(30):
        U_max = -100
        next = 0
        for neighbour in get_Neighbours(path[-1]):
            if U[neighbour] > U_max:
                U_max = U[neighbour]
                next = neighbour
        path.append(next)
        if (path[-1] == 15):
            break
    
    for i in range(len(path)):
        path[i] = (path[i] % 4, path[i] //4)
        
    print(path)
            
    
    # Reshape the 1D array into a 2D array with shape (4, 4)
    grid = U.reshape(4, 4)

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Plot the grid as a heatmap
    cax = ax.matshow(grid, cmap='viridis')

    # Add a colorbar
    plt.colorbar(cax)

    # Show the plot
    plt.show()
