import csv
import numpy as np
import pickle

NUM_VEHICLES = 16
NUM_TIMESTEPS = 108
TIMESTEP_LENGTH = 10 # seconds (real life: 300)
PARKING_STATE_INITIAL_COLUMN = 18
CONNECTED_CHARGER_INITIAL_COLUMN = 19
VEHICLE_COLUMN_COUNT = 5
CHARGING_SPOTS = [39, 40, 41] # TODO: put this in a yaml
ENTRANCE_COORDS = [14.38, 76.21]
VEHICLE_DIMS = [4.6, 1.85] # length, width
MAX_SPEED = 5

parking_states = np.zeros((NUM_VEHICLES, NUM_TIMESTEPS), dtype=int)
connected_chargers = np.zeros((NUM_VEHICLES, NUM_TIMESTEPS), dtype=int)

with open('ControllerResult.csv', newline='') as csvfile:
    rdr = csv.reader(csvfile)
    row_cnt = 0
    for row in rdr:
        if row_cnt > 0:
            for i in range(NUM_VEHICLES):
                parking_states[i][row_cnt - 1] = int(np.round(float(row[PARKING_STATE_INITIAL_COLUMN + VEHICLE_COLUMN_COUNT * i])))
                connected_chargers[i][row_cnt - 1] = int(np.round(float(row[CONNECTED_CHARGER_INITIAL_COLUMN + VEHICLE_COLUMN_COUNT * i])))
        row_cnt += 1

events = {}

for i in range(NUM_VEHICLES):
    entered = False
    charging = False
    events[i] = {}
    for t in range(NUM_TIMESTEPS):
        if not entered and parking_states[i][t] > 0:
            events[i]['enter_lot'] = t
            entered = True
        if entered and parking_states[i][t] == 0:
            events[i]['exit_lot'] = t
            entered = False

        if not charging and connected_chargers[i][t] > 0:
            events[i]['start_charging'] = t
            events[i]['charger'] = connected_chargers[i][t]
            charging = True
        if charging and connected_chargers[i][t] == 0:
            events[i]['stop_charging'] = t
            charging = False

def timestep_to_time(timestep, vehicle_id):
    return timestep * TIMESTEP_LENGTH + vehicle_id * 0.1 # tiebreaker so vehicles don't all leave at same time

vehicle_data = {}

for i in events:
    v_data = events[i]

    v_json = {}
    v_json['init_time'] = v_data['enter_lot'] * TIMESTEP_LENGTH
    v_json['init_coords'] = ENTRANCE_COORDS
    v_json['init_heading'] = -np.pi / 2 # south
    v_json['init_v'] = 0
    v_json['length'] = VEHICLE_DIMS[0]
    v_json['width'] = VEHICLE_DIMS[1]
    # differences for ev charging:
    # if parking should be chosen by NN, target_spot_index is omitted
    # for IDLE tasks, duration is replaced with end_time
    v_json['ev_charging'] = True

    task_profile = []

    parked_spot = 0
    if 'start_charging' in v_data and v_data['start_charging'] == v_data['enter_lot']: # charge right away
        # park in charger
        task_profile.append({'name': 'CRUISE', 'v_cruise': MAX_SPEED, 'target_spot_index': CHARGING_SPOTS[v_data['charger'] - 1]})
        task_profile.append({'name': 'PARK', 'target_spot_index': CHARGING_SPOTS[v_data['charger'] - 1]})
        parked_spot = CHARGING_SPOTS[v_data['charger'] - 1]
        
        # charge up
        task_profile.append({'name': 'IDLE', 'end_time': timestep_to_time(v_data['stop_charging'], i)})
    elif 'start_charging' in v_data: # park somewhere else first, then charge
        # park in non-charger
        # this is designed to be overriden by the NN
        task_profile.append({'name': 'CRUISE', 'v_cruise': MAX_SPEED})
        task_profile.append({'name': 'PARK'})

        # wait for charger
        task_profile.append({'name': 'IDLE', 'end_time': timestep_to_time(v_data['start_charging'], i)})

        # park in charger
        task_profile.append({'name': 'UNPARK'})
        task_profile.append({'name': 'CRUISE', 'v_cruise': MAX_SPEED, 'target_spot_index': CHARGING_SPOTS[v_data['charger'] - 1]})
        task_profile.append({'name': 'PARK', 'target_spot_index': CHARGING_SPOTS[v_data['charger'] - 1]})
        parked_spot = CHARGING_SPOTS[v_data['charger'] - 1]

        # charge up
        task_profile.append({'name': 'IDLE', 'end_time': timestep_to_time(v_data['stop_charging'], i)})
    else: # just enter
        # park in non-charger
        # this is designed to be overriden by the NN
        task_profile.append({'name': 'CRUISE', 'v_cruise': MAX_SPEED})
        task_profile.append({'name': 'PARK'}) 

    if 'stop_charging' in v_data and 'exit_lot' in v_data and v_data['stop_charging'] == v_data['exit_lot']: # leave right away
        task_profile.append({'name': 'UNPARK', 'target_spot_index': parked_spot})
        task_profile.append({'name': 'CRUISE', 'v_cruise': MAX_SPEED, 'target_coords': ENTRANCE_COORDS}) 
    elif 'stop_charging' in v_data and 'exit_lot' in v_data: # go somewhere else first, then leave
        # park in non-charger
        # this is designed to be overriden by the NN
        task_profile.append({'name': 'UNPARK', 'target_spot_index': parked_spot})
        task_profile.append({'name': 'CRUISE', 'v_cruise': MAX_SPEED})  
        task_profile.append({'name': 'PARK'})

        # wait to leave
        task_profile.append({'name': 'IDLE', 'end_time': timestep_to_time(v_data['exit_lot'], i)})

        # leave
        task_profile.append({'name': 'UNPARK'})
        task_profile.append({'name': 'CRUISE', 'v_cruise': MAX_SPEED, 'target_coords': ENTRANCE_COORDS})
    elif 'stop_charging' in v_data: # go somewhere else
        # park in non-charger
        # this is designed to be overriden by the NN
        task_profile.append({'name': 'UNPARK', 'target_spot_index': parked_spot})
        task_profile.append({'name': 'CRUISE', 'v_cruise': MAX_SPEED})  
        task_profile.append({'name': 'PARK'})
    elif 'exit_lot' in v_data: # just enter/exit
        task_profile.append({'name': 'IDLE', 'end_time': timestep_to_time(v_data['exit_lot'], i)})
        task_profile.append({'name': 'UNPARK', 'target_spot_index': parked_spot})
        task_profile.append({'name': 'CRUISE', 'v_cruise': MAX_SPEED, 'target_coords': ENTRANCE_COORDS})
        
    v_json['task_profile'] = task_profile
    vehicle_data[i + 1] = v_json

with open('agents_data.pickle', 'wb') as f:
    pickle.dump(vehicle_data, f)