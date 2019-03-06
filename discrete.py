import random
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use("TkAgg")  # Fixing a bug of matplotlib on MacOS
from matplotlib import pyplot as plt


# -----------------------------------------------------------------------------
class Domain:
    def __init__(self, rewards, deterministic=True, beta=0):
        """
        Init the Domain instance.
        Arguments:
        ----------
        - 'rewards': a matrix of rewards where each element represents
        the reward obtained while reaching it after a move.
        - 'deterministic' : boolean indicating if we are in a stochastic
        or deterministic setting.
        - 'beta' : parameter used to determine next state in the stochastic
        setting.
        """
        self.rewards = rewards
        self.deterministic = deterministic
        self.beta = beta
        self.discount = 0.99
        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def get_x_range(self):
        """
        Get the height of the rewards matrix.
        """
        return self.rewards.shape[0]

    def get_y_range(self):
        """
        Get the width of the rewards matrix.
        """
        return self.rewards.shape[1]

    def compute_next_state(self, state, action, stocha=True):
        """
        Compute the next state given the current state and
        an action.
        Arguments:
        ----------
        - 'state': a state defined as a position (x,y)
        - 'action': an action defined as a vector (x,y)
        - 'stocha': boolean used to get the deterministic next state
        Return:
        -------
        - The resulting state from doing the given action from
        the given state.
        """
        x, y = state
        i, j = action

        # Stochastic setting
        if not self.deterministic and stocha:
            w = np.random.uniform()
            if w > 1 - self.beta:
                return (0, 0)

        # Else : stochastic setting with w <= 1- beta
        # OR deterministic setting
        next_x = min(max(x+i, 0), self.rewards.shape[0] - 1)
        next_y = min(max(y+j, 0), self.rewards.shape[1] - 1)
        return (next_x, next_y)

    def get_reward(self, state):
        """
        Get the reward at a given cell according to the rewards matrix.
        Arguments:
        ----------
        - 'state': a state defined as a position (x,y)
        - 'action': an action defined as a vector (x,y)
        Return:
        -------
        - The value of the reward for a cell.
        """
        x, y = state
        return self.rewards[x, y]

    def compute_expected_reward(self, state, action):
        """
        Get the expected reward at a given cell according to the rewards
        matrix.
        Arguments:
        ----------
        - 'state': a state defined as a position (x,y)
        - 'action': an action defined as a vector (x,y)
        Return:
        -------
        - The value of the expected reward for a cell.
        """
        expected_reward = 0
        next_deter_state = self.compute_next_state(state, action, False)

        # Deterministic setting
        if self.deterministic:
            expected_reward = self.get_reward(next_deter_state)
        # Stochastic setting
        else:
            expected_reward = (1-self.beta)*self.get_reward(next_deter_state) \
                + self.beta*self.get_reward((0, 0))
        return expected_reward

    def get_rewards_matrix(self):
        """
        Compute a 3D matrix for all the rewards r((x,y),u).
        """
        true_rewards_matrix = np.zeros((self.get_x_range(),
                                        self.get_y_range(),
                                        len(self.actions)))

        for x in range(self.get_x_range()):
            for y in range(self.get_y_range()):
                state = (x, y)
                for u in self.actions:
                    u_index = self.actions.index(u)
                    true_rewards_matrix[x, y, u_index] = (
                        self.compute_expected_reward(state, u))
        return true_rewards_matrix

    def compute_proba(self, state, action, reached_state):
        """
        Compute the probability p(x'|x,u).
        Arguments:
        ----------
        - 'state': the current state defined as a position (x,y)
        - 'action': the action to take defined as a vector (u,w)
        - 'reached_state': the state we want to reach.
        Return:
        -------
        - The probability p(x'|x,u), that is the probability to go from a
        state x to a state x' by taking the action u.
        """
        prob = 0
        next_state = self.compute_next_state(state, action, stocha=False)

        # Deterministic case
        if self.deterministic:
            if reached_state == next_state:
                prob = 1
            else:
                prob = 0
        # Stochastic case
        else:
            if reached_state == next_state:
                prob = 1 - self.beta
            elif reached_state == (0, 0):
                prob = self.beta
            else:
                prob = 0
        return prob

    def get_probas_matrix(self):
        """
        Compute a 5D matrix for all the probabilities p((x',y')|(x,y),u).
        """
        true_prob_matrix = np.zeros((self.get_x_range(),
                                     self.get_y_range(),
                                     len(self.actions),
                                     self.get_x_range(),
                                     self.get_y_range()))

        for x in range(self.get_x_range()):
            for y in range(self.get_y_range()):
                for u in self.actions:
                    state = (x, y)
                    u_index = self.actions.index(u)
                    for x_next in range(self.get_x_range()):
                        for y_next in range(self.get_y_range()):
                            reached_state = (x_next, y_next)
                            true_prob_matrix[x, y, u_index, x_next, y_next] = (
                                self.compute_proba(state, u, reached_state))
        return true_prob_matrix


# -----------------------------------------------------------------------------
class Agent:
    def __init__(self, domain, N, policy=None):
        """
        Init the Agent instance.
        Arguments:
        ----------
        - 'domain': a domain instance.
        - 'N' : the time horizon.
        - 'policy' : a 2D matrix indicating the index of the action to take
        if there is a policy.
        """
        self.domain = domain
        self.N = N
        self.policy = policy
        self.true_rewards_matrix = None
        self.true_proba_matrix = None
        self.true_Q_matrix = None
        self.true_J_matrix = None

    def get_optimal_values(self):
        """
        Run the optimal agent that has perfect knowledge on the domain.
        """
        self.true_rewards_matrix = self.domain.get_rewards_matrix()
        self.true_proba_matrix = self.domain.get_probas_matrix()
        self.true_Q_matrix = self.compute_Q_functions(
            self.true_rewards_matrix, self.true_proba_matrix)
        self.policy = self.get_optimal_policy(self.true_Q_matrix)
        self.true_J_matrix = self.compute_value_functions(self.true_Q_matrix)

    def select_action(self, state):
        """
        Choose an action according to the current policy of the agent.
        If no policy, choose a random action.
        """
        x, y = state
        if self.policy is not None:
            return self.domain.actions[int(self.policy[x, y])]

        return self.domain.actions[random.randint(0, 3)]

    def generate_trajectories(self, t, n, state=None):
        """
        Generate a list of trajectories.
        Arguments:
        ----------
        - 't' : number of time steps considered in the trajectory.
        - 'n' : number of trajectories.
        - 'state' : initial state of the trajectory.
        Return:
        -------
        - A list of trajectories got from an initial state by following
        the policy of the agent (if one) or at random (if none).
        """
        trajectories = []

        # Generate n random trajectories
        for i in range(n):
            h_t = []
            if state is None:
                # Random initial state
                x = random.randint(0, self.domain.get_x_range()-1)
                y = random.randint(0, self.domain.get_y_range()-1)
                state = (x, y)
            # Generate one trajectory during t time steps
            for j in range(t):
                # Choose a random action
                action = self.select_action(state)
                # Compute the next state according to this action
                next_state = self.domain.compute_next_state(state, action)
                # Get the reward of the new_state
                reward = self.domain.get_reward(next_state)
                # Add to the trajectory
                h_t.append(state)
                h_t.append(action)
                h_t.append(reward)
                h_t.append(next_state)
                # Update the state
                state = next_state
            trajectories.append(h_t)
        return trajectories

    def compute_expected_returns(self):
        """
        Compute an estimate of the expected returns of a stationary policy
        for all states of a given domain.
        """
        prev_J = np.zeros(np.shape(self.domain.rewards))

        # Update the expected returns matrix for n time steps
        for i in range(self.N):
            new_J = np.zeros(np.shape(prev_J))
            # For each cell, update the expected return
            for x in range(self.domain.get_x_range()):
                for y in range(self.domain.get_y_range()):
                    state = (x, y)
                    # Choose the action according to the policy
                    action = self.select_action(state)
                    # Compute the expected reward for doing action in that state
                    exp_reward = self.domain.compute_expected_reward(
                        state, action)
                    # Get the expected return of previous matrix
                    x_next, y_next = self.domain.compute_next_state(state,
                                                                    action,
                                                                    stocha=False)
                    if self.domain.deterministic:  # Deterministic setting
                        prev_exp_return = prev_J[x_next, y_next]
                    else:  # Stochastic setting
                        prev_exp_return = (1-self.domain.beta)*prev_J[x_next, y_next] \
                            + self.domain.beta*prev_J[0, 0]
                    # Compute the new return
                    new_J[x, y] = exp_reward + \
                        (self.domain.discount * prev_exp_return)
            prev_J = new_J
        return prev_J

    def compute_Q_functions(self, rewards_matrix, proba_matrix):
        """
        Compute an estimate of the Q-function in a domain over n iterations.
        Arguments:
        ----------
        - 'rewards_matrix' : a 3D matrix for all the rewards r((x,y),u).
        - 'proba_matrix' : a 5D matrix for all the probabilities
        p((x',y')|(x,y),u).
        Return:
        -------
        - A matrix with the estimates of the Q-function for all states
        of a given domain.
        """
        prev_Q_matrix = np.zeros((self.domain.get_x_range(),
                                  self.domain.get_y_range(),
                                  len(self.domain.actions)))

        # Get a list of indexes in the proba matrix of all non-null elements
        non_null_probas = np.argwhere(proba_matrix != 0)

        # For n time steps
        for i in range(self.N):
            new_Q_matrix = np.zeros(np.shape(prev_Q_matrix))
            # Update the Q-function of each cell (x, y, u) of the matrix
            for x in range(self.domain.get_x_range()):
                for y in range(self.domain.get_y_range()):
                    for u in self.domain.actions:
                        u_index = self.domain.actions.index(u)

                        # Get the immediate reward
                        reward = rewards_matrix[x, y, u_index]

                        # Compute the sum
                        Q_sum = 0
                        # p = (x, y, u, x_next, y_next)
                        for p in non_null_probas:
                            if (p[0] == x and p[1] == y and p[2] == u_index):
                                proba = proba_matrix[p[0],
                                                     p[1], p[2], p[3], p[4]]
                                Q_sum += proba * \
                                    max(prev_Q_matrix[p[3], p[4], :])

                        # Compute the new return
                        new_Q_matrix[x, y, u_index] = reward + \
                            (self.domain.discount * Q_sum)
            prev_Q_matrix = new_Q_matrix
        return prev_Q_matrix

    def get_optimal_policy(self, Q_matrix):
        """
        Derive the optimal policy from the Q-functions for each state.
        Arguments:
        ----------
        - 'Q_matrix' : a matrix with the estimates of the Q-function for all
        states of a given domain.
        Return:
        -------
        - A tuple composed of a matrix of the optimal actions in each state,
        and a matrix of the value functions for each state.
        """
        actions_matrix = np.zeros(np.shape(self.domain.rewards))

        # For each state
        for x in range(self.domain.get_x_range()):
            for y in range(self.domain.get_y_range()):
                # Optimal action
                actions_matrix[x, y] = np.argmax(Q_matrix[x, y, :])
        return actions_matrix

    def compute_value_functions(self, Q_matrix):
        """
        Compute an estimate of the expected returns of a stationary policy
        for all states of a given domain.
        Arguments:
        ----------
        - 'Q_matrix' : a matrix with the estimates of the Q-function for all
        states of a given domain.
        Return:
        -------
        - A matrix with the estimates of the expected returns of a stationary
        policy for all states of a given domain.
        """
        J_matrix = np.zeros(np.shape(self.domain.rewards))

        # For each state
        for x in range(self.domain.get_x_range()):
            for y in range(self.domain.get_y_range()):
                # Function value
                J_matrix[x, y] = max(Q_matrix[x, y, :])
        return J_matrix


# -----------------------------------------------------------------------------
class MDP_Agent(Agent):
    def __init__(self, domain, N):
        """
        Init the MDP agent.
        """
        super().__init__(domain, N)
        self.trajectories = None  # List of trajectories
        self.rewards_matrix = None  # Matrix of estimated rewards
        self.proba_matrix = None  # Matrix of estimated probas
        self.Q_matrix = None  # Matrix of estimated Q-functions
        self.J_matrix = None  # Matrix of estimated value functions

    def run_agent(self):
        """
        Run the MDP agent to get the optimal policy from the explored domain.
        """
        # Explore the domain
        if self.domain.deterministic:
            self.trajectories = self.generate_trajectories(1000, 1)
        else:
            self.trajectories = self.generate_trajectories(1000, 1000)
        # Compute the estimates with the trajectories
        self.rewards_matrix, self.proba_matrix = self.estimate_from_trajectory()
        # Compute the Q-functions
        self.Q_matrix = self.compute_Q_functions(
            self.rewards_matrix, self.proba_matrix)
        # Compute the optimal policy
        self.policy = self.get_optimal_policy(self.Q_matrix)
        # Compute the value functions J
        self.J_matrix = self.compute_value_functions(self.Q_matrix)
        return

    def estimate_from_trajectory(self):
        """
        Given a list of trajectories, compute a matrix of estimates of all r(x,u)
        and another matrix of estimates of all p(x'|x,u).
        Also display the convergence of r and p.
        """
        count_rewards_matrix = np.zeros((self.domain.get_x_range(),
                                         self.domain.get_y_range(),
                                         len(self.domain.actions)))
        count_proba_matrix = np.zeros((self.domain.get_x_range(),
                                       self.domain.get_y_range(),
                                       len(self.domain.actions),
                                       self.domain.get_x_range(),
                                       self.domain.get_y_range()))
        count_actions_matrix = np.zeros((self.domain.get_x_range(),
                                         self.domain.get_y_range(),
                                         len(self.domain.actions)))
        est_rewards_matrix = np.zeros(np.shape(count_rewards_matrix))
        est_proba_matrix = np.zeros(np.shape(count_proba_matrix))

        # List of all the MSE over n trajectories
        rewards_error = []
        probas_error = []

        # Get the true expected rewards and probabilities
        true_rewards_matrix = self.domain.get_rewards_matrix()
        true_proba_matrix = self.domain.get_probas_matrix()

        # Compute the estimates
        for h in self.trajectories:
            i = 0
            while i < len(h)-1:
                x, y = h[i]  # Get the initial state
                # Get the index of the action
                u_index = self.domain.actions.index(h[i+1])
                reward = h[i+2]  # Get the associated reward
                x_next, y_next = h[i+3]  # Get the next state

                count_rewards_matrix[x, y, u_index] += reward
                count_proba_matrix[x, y, u_index, x_next, y_next] += 1
                count_actions_matrix[x, y, u_index] += 1

                est_rewards_matrix[x, y, u_index] = (
                    count_rewards_matrix[x, y, u_index] /
                    count_actions_matrix[x, y, u_index])
                est_proba_matrix[x, y, u_index, x_next, y_next] = (
                    count_proba_matrix[x, y, u_index, x_next, y_next] /
                    count_actions_matrix[x, y, u_index])

                # Compute the MSE on true expected rewards and probabilities
                MSE_rewards = np.square(
                    est_rewards_matrix - true_rewards_matrix).mean()
                rewards_error.append(MSE_rewards)
                MSE_probas = np.square(
                    est_proba_matrix - true_proba_matrix).mean()
                probas_error.append(MSE_probas)

                # Go to next state in the trajectory
                i = i+4

        # # Display the convergence speed of r over all trajectories
        # plt.plot(range(len(rewards_error)), rewards_error)
        # plt.xlabel('t')
        # plt.ylabel('MSE on rewards')
        # plt.show()
        # # Display the convergence speed of p over all trajectories
        # plt.plot(range(len(probas_error)), probas_error)
        # plt.xlabel('t')
        # plt.ylabel('MSE on probabilities')
        # plt.show()

        return (est_rewards_matrix, est_proba_matrix)


# -----------------------------------------------------------------------------
class Q_learning_Agent(Agent):
    def __init__(self, domain, N, replay=False, alpha=0.05, greedy=True, epsilon=0.5, tau=0.5):
        """
        Init the Q-learning agent.
        """
        super().__init__(domain, N)
        self.replay = replay
        self.alpha = alpha
        self.greedy = greedy
        self.epsilon = epsilon
        self.tau = tau
        self.trajectories = None
        self.Q_matrix = np.zeros((self.domain.get_x_range(),
                                  self.domain.get_y_range(),
                                  len(self.domain.actions)))
        self.J_matrix = None

    def run_agent(self):
        """
        Run the MDP agent to get the optimal policy from the explored domain.
        """
        Q_errors = []

        # Train over 100 episodes
        for i in range(100):
            # Explore the domain over 1000 transitions starting in (3, 3)
            self.trajectories = self.generate_trajectories(
                1000, 1, state=(3, 3))
            # Compute the Q-functions
            self.Q_matrix = self.Q_learning()
            # Compute the MSE on the Q-functions
            MSE = np.square(self.Q_matrix - self.true_Q_matrix).mean()
            Q_errors.append(MSE)
            # Compute the optimal policy
            self.policy = self.get_optimal_policy(self.Q_matrix)
            # # If epsilon decreases over time
            # self.epsilon = self.epsilon - 0.05

        # Compute the value functions J
        self.J_matrix = self.compute_value_functions(self.Q_matrix)

        # # Display the convergence of Q through the episodes
        # plt.plot(range(len(Q_errors)), Q_errors)
        # plt.xlabel('Episodes')
        # plt.ylabel('MSE on Q-functions')
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # plt.show()
        # # Print the last MSE
        # print("MSE on estimated Q after the 100 episodes : {}".format(
        #     Q_errors[-1]))
        return

    def select_action(self, state):
        """
        Choose an action according to the agent policy.
        Arguments:
        ----------
        - 'state': a state from which to select an action.
        """
        x, y = state

        if self.policy is not None:
            # Select action according to the epsilon-greedy policy
            if self.greedy:
                # With a probability of epsilon, choose random action
                if np.random.uniform() < self.epsilon:
                    return self.domain.actions[random.randint(0, 3)]
                # With probability of 1-epsilon, choose the best possible action
                else:
                    return self.domain.actions[int(self.policy[x, y])]
            # Select action according to the softmax policy
            else:
                probas = []
                # Compute the probability of being selected for each action
                for u_index in range(len(self.domain.actions)):
                    num = np.exp(self.Q_matrix[x, y, u_index] / self.tau)
                    den = sum(np.exp(self.Q_matrix[x, y, :] / self.tau))
                    probas.append(np.nan_to_num(num/den))
                # Select the action
                rand = np.random.uniform()
                for u_index, u in enumerate(self.domain.actions):
                    if rand < sum(probas[:(u_index+1)]):
                        return u

        # If no policy yet, return a random action
        return self.domain.actions[random.randint(0, 3)]

    def Q_learning(self):
        """
        Compute the Q-learning algorithm
        """
        Q_matrix = self.Q_matrix
        trajectories = []
        Q_errors = []

        # Get the true Q_matrix to display convergence
        if self.true_Q_matrix is None:
            self.get_optimal_values()

        # If agent makes experience replay
        if self.replay:
            trajectory = []
            for i in range(30000):
                # Choose at random one of the trajectories
                traj_index = random.randint(0, len(self.trajectories)-1)
                # Choose at random one sample from this trajectory
                sample_index = random.randrange(
                    0, len(self.trajectories[traj_index])-4, 4)
                # Append the sample to the infinite trajectory
                # Initial state
                trajectory.append(self.trajectories[traj_index][sample_index])
                trajectory.append(
                    self.trajectories[traj_index][sample_index+1])  # Action
                trajectory.append(
                    self.trajectories[traj_index][sample_index+2])  # Reward
                trajectory.append(
                    self.trajectories[traj_index][sample_index+3])  # Next state
            trajectories.append(trajectory)
        else:
            trajectories = self.trajectories

        for h in trajectories:
            k = 0
            while k < len(h)-1:
                x, y = h[k]  # Initial state
                u_index = self.domain.actions.index(
                    h[k+1])  # Index of the action
                reward = h[k+2]  # Associated reward
                x_next, y_next = h[k+3]  # Next state

                # Q-learning update
                Q_matrix[x, y, u_index] = (1-self.alpha)*Q_matrix[x, y, u_index] + self.alpha*(
                    reward + self.domain.discount * max(Q_matrix[x_next, y_next, :]))

                # Go to the next state in the trajectory
                k = k+4

                # Compute the MSE
                MSE = np.square(self.true_Q_matrix - Q_matrix).mean()
                Q_errors.append(MSE)

        # # Display convergence of Q
        # plt.plot(range(len(Q_errors)), Q_errors)
        # plt.xlabel('t')
        # plt.ylabel('MSE on Q-functions')
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # plt.show()

        return Q_matrix


# -----------------------------------------------------------------------------
def convert_action_to_symbol(actions_matrix):
    """
    Convert an actions matrix to a corresponding matrix of strings
    indicating the actions.
    """
    arrows_matrix = np.chararray(np.shape(actions_matrix),
                                 itemsize=6,
                                 unicode=True)
    for x in range(actions_matrix.shape[0]):
        for y in range(actions_matrix.shape[1]):
            if actions_matrix[x, y] == 0:  # Going down
                arrows_matrix[x, y] = ' down'
            elif actions_matrix[x, y] == 1:  # Going up
                arrows_matrix[x, y] = '   up'
            elif actions_matrix[x, y] == 2:  # Going right
                arrows_matrix[x, y] = 'right'
            elif actions_matrix[x, y] == 3:  # Going left
                arrows_matrix[x, y] = ' left'
    return arrows_matrix


def print_in_a_frame(message, symbol):
    size = len(message)
    print("\n")
    print(symbol * (size + 4))
    print('* {:<{}} *'.format(message, size))
    print(symbol * (size + 4))


def underline(message):
    size = len(message)
    print("")
    print('--> {:<{}}'.format(message, size))
    print('-' * (size+4))


if __name__ == "__main__":
    np.set_printoptions(precision=2)

    # Rewards matrix
    rewards = np.matrix(([-3, 1, -5, 0, 19],
                         [6, 3, 8, 9, 10],
                         [5, -8, 4, 1, -8],
                         [6, -9, 4, 19, -5],
                         [-20, -17, -4, -3, 9]))
    print("")
    print("Matrix of rewards :")
    print(rewards)
    print("")

    print("The discount factor is set to 0.99")

    # Domain instances
    deter_domain = Domain(rewards)  # Deterministic
    stocha_domain = Domain(rewards, deterministic=False, beta=0.1)  # Stocha

    # Fix the time horizon for computing Q and J
    N = 2000
    print("Time horizon of N = {}".format(N))

    print_in_a_frame("6. Q-learning in a batch setting", '=')
    underline("Deterministic domain")
    print("Agent using an epsilon-greedy policy with epsilon=0.02\n")
    q_agent_deter = Q_learning_Agent(
        deter_domain, N, replay=True, epsilon=0.02)
    q_agent_deter.run_agent()
    print("Estimate of optimal policy derived from estimated Q :")
    print(convert_action_to_symbol(q_agent_deter.policy))
    print("")
    print("Estimate of value function J for each initial state:")
    print(q_agent_deter.J_matrix)
    print("")
    print("Agent using an epsilon-greedy policy with epsilon=0.2\n")
    q_agent_deter = Q_learning_Agent(deter_domain, N, replay=True, epsilon=0.2)
    q_agent_deter.run_agent()
    print("Estimate of optimal policy derived from estimated Q :")
    print(convert_action_to_symbol(q_agent_deter.policy))
    print("")
    print("Estimate of value function J for each initial state:")
    print(q_agent_deter.J_matrix)
    print("")
    print("Agent using an epsilon-greedy policy with epsilon=0.5\n")
    q_agent_deter = Q_learning_Agent(deter_domain, N, replay=True, epsilon=0.5)
    q_agent_deter.run_agent()
    print("Estimate of optimal policy derived from estimated Q :")
    print(convert_action_to_symbol(q_agent_deter.policy))
    print("")
    print("Estimate of value function J for each initial state:")
    print(q_agent_deter.J_matrix)
    print("")

    underline("Stochastic domain")
    print("Agent using an epsilon-greedy policy with epsilon=0.02\n")
    q_agent_stocha = Q_learning_Agent(
        stocha_domain, N, replay=True, epsilon=0.02)
    q_agent_stocha.run_agent()
    print("Estimate of optimal policy derived from estimated Q :")
    print(convert_action_to_symbol(q_agent_stocha.policy))
    print("")
    print("Estimate of value function J for each initial state:")
    print(q_agent_stocha.J_matrix)
    print("")
    print("Agent using an epsilon-greedy policy with epsilon=0.2\n")
    q_agent_stocha = Q_learning_Agent(
        stocha_domain, N, replay=True, epsilon=0.2)
    q_agent_stocha.run_agent()
    print("Estimate of optimal policy derived from estimated Q :")
    print(convert_action_to_symbol(q_agent_stocha.policy))
    print("")
    print("Estimate of value function J for each initial state:")
    print(q_agent_stocha.J_matrix)
    print("")
    print("Agent using an epsilon-greedy policy with epsilon=0.5\n")
    q_agent_stocha = Q_learning_Agent(
        stocha_domain, N, replay=True, epsilon=0.5)
    q_agent_stocha.run_agent()
    print("Estimate of optimal policy derived from estimated Q :")
    print(convert_action_to_symbol(q_agent_stocha.policy))
    print("")
    print("Estimate of value function J for each initial state:")
    print(q_agent_stocha.J_matrix)
    print("")

    underline("Deterministic domain")
    print("Agent using a softmax policy with tau=1000\n")
    q_agent_deter = Q_learning_Agent(
        deter_domain, N, replay=True, greedy=False, tau=1000)
    q_agent_deter.run_agent()
    print("Estimate of optimal policy derived from estimated Q :")
    print(convert_action_to_symbol(q_agent_deter.policy))
    print("")
    print("Estimate of value function J for each initial state:")
    print(q_agent_deter.J_matrix)
    print("")
    underline("Stochastic domain")
    print("Agent using a softmax policy with tau=1000\n")
    q_agent_deter = Q_learning_Agent(
        stocha_domain, N, replay=True, greedy=False, tau=1000)
    q_agent_deter.run_agent()
    print("Estimate of optimal policy derived from estimated Q :")
    print(convert_action_to_symbol(q_agent_deter.policy))
    print("")
    print("Estimate of value function J for each initial state:")
    print(q_agent_deter.J_matrix)
    print("")
