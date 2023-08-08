# the policy will output the actions for steering, acceleration, brake, etc. This will be a neural network. In forward we pass the current state containing info about the karts
# as well as the puck and goal. The model predicts the actions our kart will take. We can use the provided sample agents as our ground truth data. 
# Since our outputs range from -1 to 1 and 0 to 1, we need to use different activation functions and our output size will need to change. I think using
# output size of 3 will work for acceleration, steering, brake, and then use sigmoid/tanh activations to achieve 0 to 1 and -1 to 1

import torch
import pickle
import random
from feature_extraction import limit_period, extract_features
import numpy as np

# policy function is the set of actions we take (steering, acceleration, etc). an optimal policy will be desired as that means our karts are playing optimally
class Policy(torch.nn.Module):
    def __init__(self, state_dim, num_actions, hidden_size): # suppose only 1 hidden layer for now
        super().__init__()
        self.linear1 = torch.nn.Linear(state_dim, hidden_size)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, num_actions)

        self.network = torch.nn.Sequential([
            torch.nn.Linear(state_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_actions)
        ])
    
    def forward(self, x):
        x = self.network(x) # apply network to x

        # now x has size equivalent to the number of controller parameters we're learning
        # suppose for now the parameters are acceleration, steering, brake
        # acceleration ranges from 0 to 1, steering ranges from -1 to 1, and brake is either 0 or 1.

        acceleration = x[:, 0] # get acceleration for each batch
        steering = x[:, 1] # get steering for each batch
        brake = x[:, 2] # get braking for each batch

        acceleration = torch.sigmoid(acceleration) # apply sigmoid function to acceleration to get output between 0 and 1
        steering = torch.tanh(steering) # apply tanh function to steering to get outpu between -1 and 1

        # for brake we first apply sigmoid to get the output between 0 and 1, and then round to either 0 or 1
        brake = torch.sigmoid(brake) # brake is float between 0 and 1
        brake = torch.round(brake) # if brake >= 0.5 then we get 1 otherwise we get 0

        # recombine the tensors
        z = torch.stack([acceleration, steering, brake], dim=1) 

        # now we have the probability for each action
        dist = torch.distributions.Categorical(z) # create a distribution
        selected_action = dist.sample() # get an action

        return selected_action

    def update(self, x, action, gamma, delta): # delta if using baseline
        '''
        x: state
        gamma: reward reduction factor (discount)
        delta: G - V # difference in value of states
        '''
        # weights = weights + alpha * gamma * delta * gradient
        alpha = .001

        action_probs = self.network(x) # first get the action probabilities
        dist = torch.distributions.Categorical(action_probs) # create a distribution
        log_prob = dist.log_prob(action) # get the log probability of the selected action

        optimizer = torch.optim.Adam(self.network.parameters(), lr = alpha)
        loss = - gamma * delta * log_prob # get the loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        


    

# we need a way to predict the value of a given state in order to update our policy. Our approximator is also something that needs to be learned.
class ValueApproximation(torch.nn.Module):
    # class to predict the value of a given state.
    # may need to include baseline
    def __init__(self, state_dims, hidden_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(state_dims, hidden_size)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, 1) # only 1 output dimension since the value of a state is a float

        self.network = torch.nn.Sequential([
            torch.nn.Linear(state_dims, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        ])
        

    def forward(self, x):
        # returns value of the current state x
        x = self.linear2(self.relu(self.linear1(x))) # feed state x through out network

        return x
    
    def update(self, x, G):
        # updates our value function approximator
        x = self.network(x)
        alpha = .001

        loss = torch.nn.MSELoss()
        loss_val = loss(x, G) 
        optimizer = torch.optim.Adam(self.network.parameters(), lr=alpha)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        
    



def REINFORCE(file_path, gamma, pi, V):
    '''
    file_path: training data
    gamma: reward reduction factor (discount)
    pi: policy we're following
    V: ValueApproximation
    '''

    # this is where we train. 

    for epoch in range(50):

        # use env to generate an episode: S0, A0, R1, S1, A1, R2, ..., S_(T-1), A_(T-1), RT
        # these episodes are in accordance with our policy pi
        # S = list, R = list, A = list

        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        simulation = random.choice(data) # randomly select a simulation from the data

        S = []
        A = []
        R = []


        for step in range(len(simulation)): # loop through each step
            current_step = simulation[step] # get the current step

            pstate = current_step['team1_state']
            soccer_state = current_step['soccer_state']
            opponent_state = current_step['team2_state']
            team_id = 1 # red is 0 and blue is 1

            features = extract_features(pstate, soccer_state, opponent_state, team_id) # get the state features
            S.append(features)

            action = current_step['actions'][0] # get the action associated with the first kart
            A.append(action)
            
            # give a positive reward for how close the puck is to their goal and a negative reward for how close it is to our goal
            #puck_to_goal_line_red = features[9] # the ninth index has to do with distance to the red team goal line
            #puck_to_goal_line_blue = features[10] # the ninth index has to do with distance to the blue team goal line
            #pos_reward = 1 / puck_to_goal_line_blue
            #neg_reward = -1 / puck_to_goal_line_red

            reward = features[8] # puck center from team_id 1 perspective. ranges from -64.5 (own goal) to 64.5 (other team goal)
            # modify reward so it's exponential towards either goal line
            if reward < 0:
                reward = -1 * (np.exp(np.abs(reward/32)) - 1)
            else:
                reward = np.exp(reward/32) - 1
        
            R.append(reward)


        G = 0 
        
        T = len(simulation)
        for t in range(T):
            for k in range(t + 1, T + 1):
                G += gamma**(k - t - 1) * R[k] # update G
            
            delta = G - V(S[t]) # calculate delta

            # update value function and policy
            V.update(S[t], G)
            pi.update(S[t], A[t], gamma**t, delta)



class Imitation(torch.nn.Module):
    def __init__(self, state_dim=17, hidden_size=32, action_size=3):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        x = self.network(x)

        acceleration = x[0] # get acceleration for each batch
        steering = x[1] # get steering for each batch
        brake = x[2] # get braking for each batch

        acceleration = torch.sigmoid(acceleration) # apply sigmoid function to acceleration to get output between 0 and 1
        steering = torch.tanh(steering) # apply tanh function to steering to get outpu between -1 and 1

        # for brake we first apply sigmoid to get the output between 0 and 1, and then round to either 0 or 1
        brake = torch.sigmoid(brake) # brake is float between 0 and 1
        brake = torch.round(brake) # if brake >= 0.5 then we get 1 otherwise we get 0

        # recombine the tensors
        z = torch.cat([acceleration.unsqueeze(0), steering.unsqueeze(0), brake.unsqueeze(0)])

        return z
    

def train(model):
    model = model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(20):
        print(epoch)

        with open("full_data.pkl", 'rb') as file:
            data = pickle.load(file)

        simulation = random.choice(data) # randomly select a simulation from the data

        for step in range(len(simulation)): # loop through each step
            current_step = simulation[step] # get the current step

            pstate = current_step['team1_state'][0]
            soccer_state = current_step['soccer_state']
            opponent_state = current_step['team2_state']
            team_id = 1 # red is 0 and blue is 1

            features = extract_features(pstate, soccer_state, opponent_state, team_id).to(device) # get the state features

            accel = current_step['actions'][0]["acceleration"]
            steer = current_step['actions'][0]["steer"]
            brake = current_step['actions'][0]["brake"]
            action = torch.stack([accel, steer, brake]).squeeze().to(device)


            optimizer.zero_grad()
            pred_action = model(features) # get predicted action
            loss = loss_function(pred_action, action) # calculate loss
            loss.backward()
            optimizer.step()