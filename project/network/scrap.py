import pickle
import torch
import data_collection
import numpy as np

data = data_collection.full_data

#print(data[0][0]["team1_state"])

def limit_period(angle):
    # turn angle into -1 to 1 
    return angle - torch.floor(angle / 2 + 0.5) * 2 

def extract_featuresV2(pstate, soccer_state, opponent_state, team_id):
    # features of ego-vehicle
    kart_front = torch.tensor(pstate['kart']['front'], dtype=torch.float32)[[0, 2]]
    kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
    kart_direction = (kart_front-kart_center) / torch.norm(kart_front-kart_center)
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

    # features of soccer 
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center-kart_center)
    kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0]) 

    kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle)/np.pi)

    # features of score-line 
    goal_line_center = torch.tensor(soccer_state['goal_line'][(team_id+1)%2], dtype=torch.float32)[:, [0, 2]].mean(dim=0)

    puck_to_goal_line = (goal_line_center-puck_center) / torch.norm(goal_line_center-puck_center)

    features = torch.tensor([kart_center[0], kart_center[1], kart_angle, kart_to_puck_angle, 
        goal_line_center[0], goal_line_center[1], kart_to_puck_angle_difference, 
        puck_center[0], puck_center[1], puck_to_goal_line[0], puck_to_goal_line[1]], dtype=torch.float32)

    return features 

for i in range(360, 730):
    print(i)
    curr_pstate = data[4][i]['team1_state'][0]
    curr_soccer_state = data[4][i]['soccer_state']
    curr_opp_state = data[4][i]['team2_state']
    curr_team_id = 1

    curr_features = extract_featuresV2(curr_pstate, curr_soccer_state, curr_opp_state, curr_team_id)

    print(curr_features[4], curr_features[7], curr_features[9], curr_features[5], curr_features[8], curr_features[10])

#print(data[100]['actions'])

# keys: ['team1_state', 'team2_state', 'soccer_state', 'actions']

# need two data files. one is the current data file which contains data from the most recent simulation and the other stores data from all simulations

# the first and 3rd action are from the same team, and 2 and 4 are from same team

# red team is team1
