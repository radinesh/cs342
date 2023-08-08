# the policy will output the actions for steering, acceleration, brake, etc. This will be a neural network. In forward we pass the current state containing info about the karts
# as well as the puck and goal. The model predicts the actions our kart will take. We can use the provided sample agents as our ground truth data. 
# Since our outputs range from -1 to 1 and 0 to 1, we need to use different activation functions and our output size will need to change. I think using
# output size of 3 will work for acceleration, steering, brake, and then use sigmoid/tanh activations to achieve 0 to 1 and -1 to 1

import torch

# policy function is the set of actions we take (steering, acceleration, etc). an optimal policy will be desired as that means our karts are playing optimally
class Policy(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size): # suppose only 1 hidden layer for now
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.linear2(self.relu(self.linear1(x))) # apply network to x

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

        return z

    def update(self, x, action, gamma, delta):
        '''
        x: state
        gamma: reward reduction factor (discount)
        delta: G - V # difference in value of states
        '''
    

# we need a way to predict the value of a given state in order to update our policy. Our approximator is also something that needs to be learned.
class ValueApproximation(torch.nn.Module):
    # class to predict the value of a given state.
    # may need to include baseline
    def __init__(self):
        super().__init__()
        # model
    

    def forward(self, x):
        # returns value of the current state x
        return None
    
    def update(self, x, G):
        # updates our value function approximator
        return None
    
def REINFORCE(env, gamma, pi, V):
    '''
    env: I think this is just our training data. Basically we need to be able to determine the reward and next state upon performing some action in our current state.
    gamma: reward reduction factor (discount)
    pi: policy we're following
    V: value approximator
    '''

    # this is where we train. 