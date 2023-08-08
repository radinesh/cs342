import pickle

def collect_data(file):
    '''
    All the data will be stored in a folder where each file in that folder is a separate simulation.
    This function will go through each simulation and create a temporary list that stores all the data for that simulation.
    Then we can add that list to another list or maybe pickle file and so at the end we'll have one file/list with all the data.
    '''
    temp_list = []

    with open(file, "rb") as f:
        try:
            while True:
                temp_list.append(pickle.load(f))
        except EOFError:
            pass

    return temp_list


def store_data(num_files):
    '''
    Each data file will be numbered from 0 onwards. So we can loop through range(num_files) to access each individual file.
    Then call collect_data(file) to get the data for that file and append it to our total list/file.
    '''
    full_data = []

    for i in range(num_files): # loop through each file indicator
        full_path = "data" + "/" + "my_data" + str(i) + ".pkl"

        current_data = collect_data(full_path) # get the current data list
        full_data.append(current_data)

    return full_data


full_data = store_data(6)

with open("full_data.pkl", "wb") as file:
    pickle.dump(full_data, file)