import numpy as np
from scene_graph_generation import SceneGraph

MODELS_INFO_FILE_PATH = "./models/models_info.xlsx"

print(1)
scene_graph = SceneGraph((4, 4))
scene_graph.LoadModelsInfo(MODELS_INFO_FILE_PATH)
for i in range(20):
    np.random.seed(1713104701)
    # while scene_graph.CreateScene(6) is False:
    #     continue
    # print(scene_graph)
    scene_graph.CreateScene(6)
    print(np.random.rand())
