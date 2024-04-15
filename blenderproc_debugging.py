import numpy as np
from scene_graph_generation import SceneGraph

print(1)
s = SceneGraph((4, 4))
for i in range(10):
    np.random.seed(1713104701)
    s.CreateScene()
