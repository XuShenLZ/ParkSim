f = open('starting_positions.json', 'w')
spawn_points = [(230.0 - 5.0*i, -170.0) for i in range(31)]
for i in range(1, 32):
    f.write('\n{\n\t"type": "vehicle.audi.a2", \
                \n\t"id": "vehicle_'+str(i)+'",'+
                '\n\t"spawn_point": ' + 
                '{"x": ' + str(spawn_points[i-1][0]) + ', ' + 
                '"y": ' + str(spawn_points[i-1][1]) + ', ' + 
                '"z": 40.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},\n\t"sensors": [{"type": "actor.pseudo.control", "id": "control"}]\n},'
    )
f.close()