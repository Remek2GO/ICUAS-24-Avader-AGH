import numpy as np

from collada import Collada

# Wczytaj plik Collada
dae = Collada('/home/vision/Documents/Repositorium/icuas24_avader/models/eggplant_2/eggplant_2.dae')


# Wypisz wszystkie owocki i ich pozycje
spheres = []
xyz = []
for node in dae.scene.nodes:
    if node.name.split(".")[0] == "Sphere":
        spheres.append(node)
        xyz.append(node.transforms[0].matrix[:3, 3])


# Wyszukaj pary owocków, które są mają te same współrzędne
pairs = []
pairs_flag = False
used_pairs = []
for i in range(len(xyz)):
    pairs_flag = False
    for j in range(i + 1, len(xyz)):
        if np.all(xyz[i] == xyz[j]):
            pairs.append((spheres[i], spheres[j]))
            pairs_flag = True
            used_pairs.append(spheres[i].name)  
            used_pairs.append(spheres[j].name)  
            break

    if not pairs_flag and spheres[i].name not in used_pairs:
        pairs.append(spheres[i])

print(dae)


dae.geometries[0].name = 'Sphere'