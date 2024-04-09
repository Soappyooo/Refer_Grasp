from __future__ import annotations
import numpy as np
import pandas as pd
import pickle
import os
import re
import functools

# TODO: 不同渲染背景

# 对于一个表达式，其结构为[location, attribute, objectRefer]，其中location为位置关系，attribute为属性，objectRefer为物体指代。
# 生成节点的思路是: 随机生成位置关系并检查位置是否可用 》 随机生成属性和物体指代并检查该位置的forbiddenExpressionComponentsMap 》
#                 生成节点，更新forbiddenExpressionComponentsMap，生成表达式

# 此文件包含SceneGraph类，用于生成场景图；ObjectNode类，用于生成场景图中的物体节点；ExpressionComponents类，用于存储表达式结构。
# SceneGraph类的CreateTree方法目前生成一个简单场景树，还没有加入表达式指代是否模糊的检查。
# ObjectNode类的实例表示场景图中的一个物体节点，包含物体的名称、类别、坐标、颜色、大小、表达式结构、父节点、子节点等信息。
# 对于多个物体位置关系(如in the middle of)，指代物体为父节点，其它相关节点为子节点。这些子节点的表达式结构为空列表。
# ExpressionComponents类的实例表示表达式结构，包含位置关系、属性、物体指代等信息。
# 如expressionComponents.location.relativeLocationToSingleObject.front表示位置关系为相对于单个物体的前方。
# 其它注释都用英文写了。


# obj file name, obj name, obj category, obj color
test_obj_list = [
    ["obj1", "apple", "fruit", "red"],
    ["obj2", "banana", "fruit", "yellow"],
    ["obj3", "orange", "fruit", "orange"],
    ["obj4", "apple", "fruit", "green"],
    ["obj5", "pear", "fruit", "yellow"],
    ["obj6", "strawberry", "fruit", "red"],
    ["obj7", "grape", "fruit", "purple"],
    ["obj8", "watermelon", "fruit", "green"],
    ["obj9", "tomato", "fruit", "red"],
    ["obj10", "peach", "fruit", "red"],
    ["obj11", "lemon", "fruit", "yellow"],
    ["obj12", "mango", "fruit", "yellow"],
    ["obj13", "pineapple", "fruit", "yellow"],
    ["obj14", "box", "container", "red"],
    ["obj15", "box", "container", "yellow"],
    ["obj16", "box", "container", "green"],
    ["obj17", "box", "container", "blue"],
    ["obj18", "box", "container", "purple"],
    ["obj19", "box", "container", "orange"],
    ["obj20", "can", "container", "red"],
    ["obj21", "can", "container", "yellow"],
    ["obj22", "can", "container", "green"],
    ["obj23", "can", "container", "blue"],
    ["obj24", "can", "container", "purple"],
    ["obj25", "can", "container", "orange"],
    ["obj26", "bottle", "container", "red"],
    ["obj27", "bottle", "container", "yellow"],
    ["obj28", "bottle", "container", "green"],
    ["obj29", "bottle", "container", "blue"],
    ["obj30", "bottle", "container", "purple"],
    ["obj31", "bottle", "container", "orange"],
    ["obj32", "screw", "tool", "red"],
    ["obj33", "screw", "tool", "yellow"],
    ["obj34", "screw", "tool", "green"],
    ["obj35", "screw", "tool", "blue"],
    ["obj36", "screw", "tool", "purple"],
    ["obj37", "screw", "tool", "orange"],
    ["obj38", "hammer", "tool", "red"],
    ["obj39", "hammer", "tool", "yellow"],
    ["obj40", "hammer", "tool", "green"],
    ["obj41", "hammer", "tool", "blue"],
    ["obj42", "hammer", "tool", "purple"],
    ["obj43", "hammer", "tool", "orange"],
    ["obj44", "saw", "tool", "red"],
    ["obj45", "saw", "tool", "yellow"],
    ["obj46", "saw", "tool", "green"],
    ["obj47", "saw", "tool", "blue"],
    ["obj48", "saw", "tool", "purple"],
    ["obj49", "saw", "tool", "orange"],
    ["obj50", "knife", "tableware", "red"],
    ["obj51", "knife", "tableware", "yellow"],
    ["obj52", "knife", "tableware", "green"],
    ["obj53", "knife", "tableware", "blue"],
    ["obj54", "knife", "tableware", "purple"],
    ["obj55", "knife", "tableware", "orange"],
    ["obj56", "fork", "tableware", "red"],
    ["obj57", "fork", "tableware", "yellow"],
    ["obj58", "fork", "tableware", "green"],
    ["obj59", "fork", "tableware", "blue"],
    ["obj60", "fork", "tableware", "purple"],
    ["obj61", "fork", "tableware", "orange"],
    ["obj62", "spoon", "tableware", "red"],
    ["obj63", "spoon", "tableware", "yellow"],
    ["obj64", "spoon", "tableware", "green"],
    ["obj65", "spoon", "tableware", "blue"],
    ["obj66", "spoon", "tableware", "purple"],
    ["obj67", "spoon", "tableware", "orange"],
]


class SceneGraph:
    """
    [WIP] A scene graph with object nodes, map and expressions.
    """

    def __init__(self, mapSize: tuple[int, int]) -> None:
        """
        Create an empty scene with a map of size `mapSize`.\n

        Args:
            mapSize (tuple[int, int]): The size of the scene map (2D array).
        """
        self.expressionComponents = ExpressionComponents()
        self.objectNodes: list[ObjectNode] = []
        self.mapSize = mapSize
        self.map = np.empty(mapSize, dtype=ObjectNode)
        self.forbiddenExpressionComponentsRegionMap = np.empty(mapSize, dtype=object)
        self.forbiddenExpressionComponentsGlobal: list[dict] = []
        for i in range(mapSize[0]):
            for j in range(mapSize[1]):
                self.forbiddenExpressionComponentsRegionMap[i, j] = []
        self.referringExpressionStructures: dict[ObjectNode, list] = dict()
        self.dataset: list = []
        self.retryTimes = 50

    def CreateScene(self, minObjectsNumber: int) -> bool:
        retry = self.retryTimes
        self.ClearScene()
        while True:
            self.CreateTree()
            if len(self.objectNodes) >= minObjectsNumber:
                return True
            retry -= 1
            if retry == 0:
                return False

    def ClearScene(self):
        """
        Reset the scene graph to an empty scene.
        """
        self.objectNodes.clear()
        self.map = np.empty(self.mapSize, dtype=ObjectNode)
        self.forbiddenExpressionComponentsRegionMap = np.empty(self.mapSize, dtype=list)
        for i in range(self.mapSize[0]):
            for j in range(self.mapSize[1]):
                self.forbiddenExpressionComponentsRegionMap[i, j] = []
        self.referringExpressionStructures.clear()

    def AddBaseObjectNode(
        self,
        obj: str,
        name: str,
        category: str,
        coordinate: tuple[int, int],
        color: str,
        size: float,
        referringExpressionStructure: list,
        order: int = -1,
    ) -> ObjectNode:
        baseObjectNode = ObjectNode(self, obj, name, category, coordinate, color, size, referringExpressionStructure, order=order)
        return baseObjectNode

    def GetRandomComponents(self) -> list[int]:
        """
        Get a list of random `[location, attribute, objectRefer]` components.

        Returns:
            list[int]: A list of random `[location, attribute, objectRefer]` components.
        """
        return [
            self.expressionComponents.location.GetRandomComponent(),
            self.expressionComponents.attribute.GetRandomComponent(),
            self.expressionComponents.objectRefer.GetRandomComponent(),
        ]

    def LoadModelsInfo(self, fileName: str):
        """
        Load models information from a supported file. File format should be `.xlsx` for now.\n

        Args:
            fileName (str): The file name of the dataset.

        Raises:
            SceneGraphGenerationError: If the file format is not supported.
        """
        if fileName.endswith(".xlsx"):
            self.dataset = pd.read_excel(fileName).values.tolist()
        else:
            raise SceneGraphGenerationError("Dataset file format not supported")

    def CreateTree(self) -> bool:
        """
        [WIP] Create a tree structure with a base object node and some child object nodes.\n
        The base object node should not have relative location to single object or indirect attribute since it has no parent.
        If the base object node has relative location to multiple objects, the objects related to the base object node are generated as its children.
        If the base object node has absolute location or random location, the base object node is generated at a possible location.\n
        If the base object node has attribute 'notColor', the child will have the specified color.\n
        Then, a child object node with relative location to single object (the base object node) is generated. \n
        Will consider forbidden expression components later.
        """

        while True:
            baseObjectNodeComponents = self.GetRandomComponents()
            # cannot use relative location to single object or indirect attribute for base object
            if (
                baseObjectNodeComponents[0] not in self.expressionComponents.location.relativeLocationToSingleObject.allComponents
                and baseObjectNodeComponents[1] not in self.expressionComponents.attribute.indirectRefer.allComponents
            ):
                break
            # consider notColor attribute
            if (
                baseObjectNodeComponents[0] in self.expressionComponents.location.absoluteLocation.allComponents
                or baseObjectNodeComponents[0] == self.expressionComponents.location.randomLocation
            ) and baseObjectNodeComponents[1] == self.expressionComponents.attribute.indirectRefer.notColor:

                baseObjectNodeComponents[0] = self.expressionComponents.location.randomLocation
                # cannot use ambiguous object refer
                if baseObjectNodeComponents[2] == self.expressionComponents.objectRefer.ambiguous:
                    baseObjectNodeComponents[2] = np.random.choice(
                        [self.expressionComponents.objectRefer.name, self.expressionComponents.objectRefer.category]
                    )
                break
        order = np.random.choice(4) + 1  # for row order
        possibleCoordinatesList = self.GetLocationCoordinatesFromLocationComponent(baseObjectNodeComponents[0], order=order)
        if possibleCoordinatesList == []:
            # raise SceneGraphGenerationError("No possible coordinates for base object node with location component " +
            #                                 self.expressionComponents.GetComponentName(baseObjctNodeComponents[0]))
            return False

        retry = self.retryTimes
        while True:
            retryFlag = False

            retry -= 1
            if retry == 0:
                return False

            if baseObjectNodeComponents[0] in self.expressionComponents.location.relativeLocationToMultipleObjects.allComponents:
                coordinates = possibleCoordinatesList[np.random.choice(len(possibleCoordinatesList))]
                # if middle, randomly choose 3 objects
                if baseObjectNodeComponents[0] == self.expressionComponents.location.relativeLocationToMultipleObjects.middle:
                    baseObj = self.dataset[np.random.choice(len(self.dataset))]
                    if (
                        self.CheckForbiddenExpressions(
                            baseObj, baseObjectNodeComponents[0], baseObjectNodeComponents[1], baseObjectNodeComponents[2], coordinates[0]
                        )
                        is False
                    ):
                        retryFlag = True
                        continue
                    else:
                        leftObj = self.dataset[np.random.choice(len(self.dataset))]
                    if self.CheckForbiddenExpressions(leftObj, None, None, None, coordinates[1]) is False:
                        retryFlag = True
                        continue
                    else:
                        rightObj = self.dataset[np.random.choice(len(self.dataset))]
                    if self.CheckForbiddenExpressions(rightObj, None, None, None, coordinates[2]) is False:
                        retryFlag = True
                        continue
                    else:
                        baseObjectNode = self.AddBaseObjectNode(
                            baseObj[0], baseObj[1], baseObj[2], coordinates[0], baseObj[3], 1, baseObjectNodeComponents
                        )
                        baseObjectNode.AddChild(leftObj[0], leftObj[1], leftObj[2], coordinates[1], leftObj[3], 1, [])
                        baseObjectNode.AddChild(rightObj[0], rightObj[1], rightObj[2], coordinates[2], rightObj[3], 1, [])
                # if row order, randomly choose 4 same objects
                elif baseObjectNodeComponents[0] == self.expressionComponents.location.relativeLocationToMultipleObjects.rowOrder:
                    baseObj = self.dataset[np.random.choice(len(self.dataset))]
                    baseObjectNodeComponents[2] = np.random.choice(
                        [self.expressionComponents.objectRefer.name, self.expressionComponents.objectRefer.category]
                    )
                    for i in range(1, len(coordinates)):
                        if (
                            self.CheckForbiddenExpressions(
                                baseObj, baseObjectNodeComponents[0], baseObjectNodeComponents[1], baseObjectNodeComponents[2], coordinates[i]
                            )
                            is False
                        ):
                            retryFlag = True
                            break
                    if retryFlag is False:
                        baseObjectNode = self.AddBaseObjectNode(
                            baseObj[0], baseObj[1], baseObj[2], coordinates[0], baseObj[3], 1, baseObjectNodeComponents, order=order
                        )
                        for i in range(1, len(coordinates)):
                            baseObjectNode.AddChild(baseObj[0], baseObj[1], baseObj[2], coordinates[i], baseObj[3], 1, [])
                    else:
                        continue
            # if absolute location, randomly choose an object
            elif (
                baseObjectNodeComponents[0] in self.expressionComponents.location.absoluteLocation.allComponents
                or baseObjectNodeComponents[0] == self.expressionComponents.location.randomLocation
            ) and baseObjectNodeComponents[1] != self.expressionComponents.attribute.indirectRefer.notColor:

                coordinate = possibleCoordinatesList[np.random.choice(len(possibleCoordinatesList))]
                baseObj = self.dataset[np.random.choice(len(self.dataset))]
                # for absolute location, there should not be an object already satisfied the expression
                possibleCoordinatesToCheck = self.GetLocationCoordinatesFromLocationComponent(baseObjectNodeComponents[0], None, filter=False)
                attributeDict = {
                    self.expressionComponents.attribute.directRefer.color: baseObj[3],
                    self.expressionComponents.attribute.randomAttribute: True,
                }
                objectReferDict = {
                    self.expressionComponents.objectRefer.name: baseObj[1],
                    self.expressionComponents.objectRefer.category: baseObj[2],
                    self.expressionComponents.objectRefer.ambiguous: True,
                }
                for coordinateToCheck in possibleCoordinatesToCheck:
                    if self.map[coordinateToCheck] is not None and self.map[coordinateToCheck].SatisfyExpression(
                        {
                            baseObjectNodeComponents[1]: attributeDict[baseObjectNodeComponents[1]],
                            baseObjectNodeComponents[2]: objectReferDict[baseObjectNodeComponents[2]],
                        }
                    ):
                        retryFlag = True
                        break

                if (
                    self.CheckForbiddenExpressions(
                        baseObj, baseObjectNodeComponents[0], baseObjectNodeComponents[1], baseObjectNodeComponents[2], coordinate
                    )
                    is False
                ):
                    retryFlag = True
                    continue
                if retryFlag is False:
                    baseObjectNode = self.AddBaseObjectNode(baseObj[0], baseObj[1], baseObj[2], coordinate, baseObj[3], 1, baseObjectNodeComponents)
                else:
                    continue
            # if notColor, create a random object with the specified color and one not specified color
            elif baseObjectNodeComponents[1] == self.expressionComponents.attribute.indirectRefer.notColor:
                childObj = self.dataset[np.random.choice(len(self.dataset))]
                possibleBaseObjs = []
                for line in self.dataset:
                    if line[3] != childObj[3]:
                        if baseObjectNodeComponents[2] == self.expressionComponents.objectRefer.name and line[1] == childObj[1]:
                            possibleBaseObjs.append(line)
                        elif baseObjectNodeComponents[2] == self.expressionComponents.objectRefer.category and line[2] == childObj[2]:
                            possibleBaseObjs.append(line)
                if possibleBaseObjs == []:
                    retryFlag = True
                    continue

                baseObj = possibleBaseObjs[np.random.choice(len(possibleBaseObjs))]
                coordinate = possibleCoordinatesList[np.random.choice(len(possibleCoordinatesList))]
                coordinate2 = possibleCoordinatesList[np.random.choice(len(possibleCoordinatesList))]
                if (
                    self.CheckForbiddenExpressions(
                        baseObj, baseObjectNodeComponents[0], baseObjectNodeComponents[1], baseObjectNodeComponents[2], coordinate
                    )
                    is False
                    or self.CheckForbiddenExpressions(
                        childObj, baseObjectNodeComponents[0], baseObjectNodeComponents[1], baseObjectNodeComponents[2], coordinate2
                    )
                    is False
                ):
                    retryFlag = True
                    continue

                for objectNode in self.objectNodes:
                    if objectNode.SatisfyExpression({baseObjectNodeComponents[0]: baseObj[1], baseObjectNodeComponents[1]: childObj[3]}):
                        retryFlag = True
                        break

                if retryFlag is False:
                    baseObjectNode = self.AddBaseObjectNode(baseObj[0], baseObj[1], baseObj[2], coordinate, baseObj[3], 1, baseObjectNodeComponents)
                    baseObjectNode.AddChild(childObj[0], childObj[1], childObj[2], coordinate2, childObj[3], 1, [])

            else:
                raise SceneGraphGenerationError("baseObjctNodeComponents[0] is not a valid location component")

            if retryFlag is False:
                break

        # create child
        # TODO: child of order expression node
        if baseObjectNodeComponents[0] == self.expressionComponents.location.relativeLocationToMultipleObjects.rowOrder:
            return True

        while True:
            childObjectNodeComponents = self.GetRandomComponents()
            # only use relative location to single object for child object or same color
            if (
                childObjectNodeComponents[0] in self.expressionComponents.location.relativeLocationToSingleObject.allComponents
                and childObjectNodeComponents[1] not in self.expressionComponents.attribute.indirectRefer.allComponents
            ):
                possibleCoordinatesList = self.GetLocationCoordinatesFromLocationComponent(childObjectNodeComponents[0], baseObjectNode)
                if possibleCoordinatesList != []:
                    break
            elif childObjectNodeComponents[1] == self.expressionComponents.attribute.indirectRefer.sameColor:
                childObjectNodeComponents[0] = self.expressionComponents.location.randomLocation
                possibleCoordinatesList = self.GetLocationCoordinatesFromLocationComponent(childObjectNodeComponents[0], baseObjectNode)
                if possibleCoordinatesList != []:
                    break

        # for same color
        if childObjectNodeComponents[1] == self.expressionComponents.attribute.indirectRefer.sameColor:
            possibleObjs = []
            for line in self.dataset:
                if line[3] == baseObjectNode.color and line[0] != baseObjectNode.obj_id:
                    possibleObjs.append(line)
            if possibleObjs == []:
                return False

            retry = self.retryTimes
            while True:
                childObj = possibleObjs[np.random.choice(len(possibleObjs))]
                coordinate = possibleCoordinatesList[np.random.choice(len(possibleCoordinatesList))]
                if (
                    self.CheckForbiddenExpressions(
                        childObj, childObjectNodeComponents[0], childObjectNodeComponents[1], childObjectNodeComponents[2], coordinate
                    )
                    is False
                ):
                    retryFlag = True

                if retryFlag is False:
                    baseObjectNode.AddChild(childObj[0], childObj[1], childObj[2], coordinate, childObj[3], 1, childObjectNodeComponents)
                    return True
                else:
                    retry -= 1
                    if retry == 0:
                        return False

        # for direct color
        # TODO: check forbidden expression components
        retry = self.retryTimes
        while True:

            retry -= 1
            if retry == 0:
                return False

            coordinate = possibleCoordinatesList[np.random.choice(len(possibleCoordinatesList))]
            childObj = self.dataset[np.random.choice(len(self.dataset))]
            retryFlag = False

            objectReferDict = {
                self.expressionComponents.objectRefer.name: childObj[1],
                self.expressionComponents.objectRefer.category: childObj[2],
                self.expressionComponents.objectRefer.ambiguous: True,
            }

            # for near location, the parent and child should not be the same object
            if childObjectNodeComponents[0] == self.expressionComponents.location.relativeLocationToSingleObject.near:
                if childObj[0] == baseObj[0]:
                    retryFlag = True
                    continue

                # for near location, there should not be an object already satisfied the expression
                coordinatesAroundBaseObject = self.GetLocationCoordinatesFromLocationComponent(
                    childObjectNodeComponents[0], baseObjectNode, filter=False
                )
                for coordinateAroundBaseObject in coordinatesAroundBaseObject:
                    if self.map[coordinateAroundBaseObject] is not None and self.map[coordinateAroundBaseObject].SatisfyExpression(
                        {
                            self.expressionComponents.attribute.directRefer.color: childObj[3],
                            childObjectNodeComponents[2]: objectReferDict[childObjectNodeComponents[2]],
                        }
                    ):
                        retryFlag = True
                        break

            if (
                self.CheckForbiddenExpressions(
                    childObj, childObjectNodeComponents[0], childObjectNodeComponents[1], childObjectNodeComponents[2], coordinate
                )
                is False
            ):
                retryFlag = True
                continue

            if retryFlag is False:
                break

        baseObjectNode.AddChild(childObj[0], childObj[1], childObj[2], coordinate, childObj[3], 1, childObjectNodeComponents)
        return True

    def GetLocationCoordinatesFromLocationComponent(
        self, locationComponent: int, parentObjectNode: ObjectNode = None, order: int = None, filter=True
    ) -> list[tuple[int, int]] | list[list[tuple[int, int]]]:
        """
        For a given `locationComponent`, return a list of coordinates that satisfy the `locationComponent` and filter the result with scene map.\n
        Specifically, if `locationComponent` is relative location to single object,
        return a list of coordinates that satisfy the `locationComponent` relative to `parentObjectNode`.
        If `locationComponent` is relative location to multiple objects, return a 2D list of coordinates.
        For each row, the first coordinate indicates referred location,
        and the rest indicate the relavant objects' coordinates. Require `order` if `locationComponent` is `relativeLocationToMultipleObjects.rowOrder`.\n
        If `locationComponent` is absolute location, return a list with only one coordinate that satisfy the `locationComponent`.\n

        Args:
            locationComponent (int): An attribute of `self.expressionComponents.location`
            parentObjectNode (ObjectNode, optional): The parent object node for `relativeLocationToSingleObject`. Defaults to None.
            order (int, optional): The order of the objects for `relativeLocationToMultipleObjects.rowOrder`. Defaults to None.
            filter (bool, optional): Whether to filter out coordinates which already have objects. Defaults to True.
        Raises:
            SceneGraphGenerationError: _description_

        Returns:
            list[tuple[int, int]] or list[list[tuple[int, int]]]: for `relativeLocationToSingleObject` and `absoluteLocation`,
            return a list of coordinates; for `relativeLocationToMultipleObjects`, return a 2D list of coordinates.
            If no coordinate satisfies the `locationComponent`, return an empty list.
        """
        # TODO: revise if more location components are added
        coordinateList = []
        if locationComponent in self.expressionComponents.location.relativeLocationToSingleObject.allComponents and parentObjectNode is None:
            raise SceneGraphGenerationError("locationComponent is relative location to single object, but parentObjectNode is None")
        elif locationComponent == self.expressionComponents.location.relativeLocationToMultipleObjects.rowOrder and order is None:
            raise SceneGraphGenerationError("locationComponent is relative location to multiple objects (requires order), but order is None")

        if locationComponent in self.expressionComponents.location.relativeLocationToSingleObject.allComponents:
            location_mapping = {
                self.expressionComponents.location.relativeLocationToSingleObject.front: (1, 0),
                self.expressionComponents.location.relativeLocationToSingleObject.rear: (-1, 0),
                self.expressionComponents.location.relativeLocationToSingleObject.left: (0, -1),
                self.expressionComponents.location.relativeLocationToSingleObject.right: (0, 1),
                self.expressionComponents.location.relativeLocationToSingleObject.frontLeft: (1, -1),
                self.expressionComponents.location.relativeLocationToSingleObject.frontRight: (1, 1),
                self.expressionComponents.location.relativeLocationToSingleObject.rearLeft: (-1, -1),
                self.expressionComponents.location.relativeLocationToSingleObject.rearRight: (-1, 1),
            }

            # for front, rear, left, right, frontLeft, frontRight, rearLeft, rearRight
            for location_component, adjustment in location_mapping.items():
                if locationComponent == location_component:
                    coordinate = (parentObjectNode.coordinate[0] + adjustment[0], parentObjectNode.coordinate[1] + adjustment[1])
                    if self.IsCoordinateValid(coordinate):
                        coordinateList.append(coordinate)
            # for near
            if locationComponent == self.expressionComponents.location.relativeLocationToSingleObject.near:
                for adjustment in location_mapping.values():
                    coordinate = (parentObjectNode.coordinate[0] + adjustment[0], parentObjectNode.coordinate[1] + adjustment[1])
                    if self.IsCoordinateValid(coordinate):
                        coordinateList.append(coordinate)

        elif locationComponent in self.expressionComponents.location.relativeLocationToMultipleObjects.allComponents:
            match locationComponent:
                case self.expressionComponents.location.relativeLocationToMultipleObjects.middle:
                    for i in range(self.mapSize[0]):
                        for j in range(1, self.mapSize[1] - 1):
                            coordinates = [(i, j), (i, j - 1), (i, j + 1)]
                            coordinateList.append(coordinates)
                # TODO: consider 4 objects in a row for now
                case self.expressionComponents.location.relativeLocationToMultipleObjects.rowOrder:
                    objectsNumberInRow = 4
                    if order <= 0 or order > objectsNumberInRow:
                        raise SceneGraphGenerationError("order should be in range [1, objectsNumberInRow]")
                    for i in range(self.mapSize[0]):
                        for j in range(order - 1, self.mapSize[1] - objectsNumberInRow + order):
                            coordinates = [(i, j)]
                            for k in range(objectsNumberInRow):
                                coordinates.append((i, j + k - order + 1)) if k - order + 1 != 0 else None
                            coordinateList.append(coordinates)

        elif locationComponent in self.expressionComponents.location.absoluteLocation.allComponents:
            match locationComponent:
                case self.expressionComponents.location.absoluteLocation.frontmost:
                    for i in range(self.mapSize[1]):
                        coordinateList.append((self.mapSize[0] - 1, i))
                case self.expressionComponents.location.absoluteLocation.rearmost:
                    for i in range(self.mapSize[1]):
                        coordinateList.append((0, i))
                case self.expressionComponents.location.absoluteLocation.leftmost:
                    for i in range(self.mapSize[0]):
                        coordinateList.append((i, 0))
                case self.expressionComponents.location.absoluteLocation.rightmost:
                    for i in range(self.mapSize[0]):
                        coordinateList.append((i, self.mapSize[1] - 1))
                case self.expressionComponents.location.absoluteLocation.frontLeftmost:
                    coordinateList.append((self.mapSize[0] - 1, 0))
                case self.expressionComponents.location.absoluteLocation.frontRightmost:
                    coordinateList.append((self.mapSize[0] - 1, self.mapSize[1] - 1))
                case self.expressionComponents.location.absoluteLocation.rearLeftmost:
                    coordinateList.append((0, 0))
                case self.expressionComponents.location.absoluteLocation.rearRightmost:
                    coordinateList.append((0, self.mapSize[1] - 1))

        elif locationComponent == self.expressionComponents.location.randomLocation:
            for i in range(self.mapSize[0]):
                for j in range(self.mapSize[1]):
                    coordinateList.append((i, j))

        else:
            raise SceneGraphGenerationError("locationComponent is not a valid location component")

        # filter out coordinates that are occupied
        filteredPossibleCoordinatesList = coordinateList.copy()
        if filter is True:
            if locationComponent in self.expressionComponents.location.relativeLocationToMultipleObjects.allComponents:
                for possibleCoordinatesGroup in coordinateList:
                    for coordinate in possibleCoordinatesGroup:
                        if self.map[coordinate] is not None:
                            filteredPossibleCoordinatesList.remove(possibleCoordinatesGroup)
                            break
            else:
                for coordinate in coordinateList:
                    if self.map[coordinate] is not None:
                        filteredPossibleCoordinatesList.remove(coordinate)

        return filteredPossibleCoordinatesList

    def IsCoordinateValid(self, coordinate: tuple[int, int]) -> bool:
        """
        Check if the given `coordinate` is within the scene map.

        Args:
            coordinate (tuple[int, int]): The coordinate to check.

        Returns:
            bool: True if the `coordinate` is within the scene map, False otherwise.
        """
        return coordinate[0] >= 0 and coordinate[0] < self.mapSize[0] and coordinate[1] >= 0 and coordinate[1] < self.mapSize[1]

    def GetExpression(self, objectNode: ObjectNode, is_parent: bool = False, get_parent_color: bool = True) -> str | tuple[str, bool]:

        def GetAbsoluteLocationWords(locationComponent: int) -> str:
            if locationComponent == self.expressionComponents.location.absoluteLocation.frontmost:
                absoluteLocationWords = [
                    "at the front",
                    "at front",
                    "frontmost",
                    "frontmost in the scene",
                    "in the front",
                    "in front",
                    "closest",
                    "the closest",
                    "at the forefront",
                    "in the front of the scene",
                ]
            if locationComponent == self.expressionComponents.location.absoluteLocation.rearmost:
                absoluteLocationWords = [
                    "at the back",
                    "at back",
                    "rearmost",
                    "rearmost in the scene",
                    "in the back",
                    "in back",
                    "furthest",
                    "the furthest",
                    "at the rear",
                    "in the back of the scene",
                ]
            if locationComponent == self.expressionComponents.location.absoluteLocation.leftmost:
                absoluteLocationWords = [
                    "at the left",
                    "at left",
                    "leftmost",
                    "leftmost in the scene",
                    "in the left",
                    "in left",
                    "at the leftmost",
                    "in the leftmost of the scene",
                ]
            if locationComponent == self.expressionComponents.location.absoluteLocation.rightmost:
                absoluteLocationWords = [
                    "at the right",
                    "at right",
                    "rightmost",
                    "rightmost in the scene",
                    "in the right",
                    "in right",
                    "at the rightmost",
                    "in the rightmost of the scene",
                ]
            if locationComponent == self.expressionComponents.location.absoluteLocation.frontLeftmost:
                absoluteLocationWords = [
                    "at the front left",
                    "at front left",
                    "front leftmost",
                    "front leftmost in the scene",
                    "in the front left",
                    "in front left",
                    "at the front leftmost",
                    "in the front leftmost of the scene",
                    "at the left front",
                    "at left front",
                    "left frontmost",
                    "left frontmost in the scene",
                    "in the left front",
                    "in left front",
                    "at the left frontmost",
                    "in the left frontmost of the scene",
                    "in the front left corner",
                    "in front left corner",
                    "at the front left corner",
                    "in left and front" "in the left front corner",
                    "in left front corner",
                    "at the left front corner",
                    "in front and left",
                ]
            if locationComponent == self.expressionComponents.location.absoluteLocation.frontRightmost:
                absoluteLocationWords = [
                    "at the front right",
                    "at front right",
                    "front rightmost",
                    "front rightmost in the scene",
                    "in the front right",
                    "in front right",
                    "at the front rightmost",
                    "in the front rightmost of the scene",
                    "at the right front",
                    "at right front",
                    "right frontmost",
                    "right frontmost in the scene",
                    "in the right front",
                    "in right front",
                    "at the right frontmost",
                    "in the right frontmost of the scene",
                    "in the front right corner",
                    "in front right corner",
                    "at the front right corner",
                    "in right and front",
                    "in the right front corner",
                    "in right front corner",
                    "at the right front corner",
                    "in front and right",
                ]
            if locationComponent == self.expressionComponents.location.absoluteLocation.rearLeftmost:
                absoluteLocationWords = [
                    "at the rear left",
                    "at rear left",
                    "rear leftmost",
                    "rear leftmost in the scene",
                    "in the rear left",
                    "in rear left",
                    "at the rear leftmost",
                    "in the rear leftmost of the scene",
                    "at the left rear",
                    "at left rear",
                    "left rearmost",
                    "left rearmost in the scene",
                    "in the left rear",
                    "in left rear",
                    "at the left rearmost",
                    "in the left rearmost of the scene",
                    "in the rear left corner",
                    "in rear left corner",
                    "at the rear left corner",
                    "in left and rear",
                    "in left and back",
                    "in the left rear corner",
                    "in left rear corner",
                    "at the left rear corner",
                    "in rear and left",
                    "in back and left",
                ]
            if locationComponent == self.expressionComponents.location.absoluteLocation.rearRightmost:
                absoluteLocationWords = [
                    "at the rear right",
                    "at rear right",
                    "rear rightmost",
                    "rear rightmost in the scene",
                    "in the rear right",
                    "in rear right",
                    "at the rear rightmost",
                    "in the rear rightmost of the scene",
                    "at the right rear",
                    "at right rear",
                    "right rearmost",
                    "right rearmost in the scene",
                    "in the right rear",
                    "in right rear",
                    "at the right rearmost",
                    "in the right rearmost of the scene",
                    "in the rear right corner",
                    "in rear right corner",
                    "at the rear right corner",
                    "in right and rear",
                    "in right and back",
                    "in the right rear corner",
                    "in right rear corner",
                    "at the right rear corner",
                    "in rear and right",
                    "in back and right",
                ]
            return absoluteLocationWords

        if is_parent:
            expression = "the "
            colorRefer = np.random.choice(
                [self.expressionComponents.attribute.directRefer.color, self.expressionComponents.attribute.randomAttribute]
            )
            objectRefer = np.random.choice(
                [
                    self.expressionComponents.objectRefer.name,
                    self.expressionComponents.objectRefer.category,
                ]
            )
            referDict = {}
            if colorRefer == self.expressionComponents.attribute.directRefer.color and get_parent_color:
                referDict[colorRefer] = objectNode.color
                expression += objectNode.color + " "
            if objectRefer == self.expressionComponents.objectRefer.name:
                referDict[objectRefer] = objectNode.name
                expression += objectNode.name
            elif objectRefer == self.expressionComponents.objectRefer.category:
                referDict[objectRefer] = objectNode.category
                expression += objectNode.category

            isLongExpression = False
            for node in self.objectNodes:
                if node != objectNode and node.SatisfyExpression(referDict):
                    isLongExpression = True
                    break
            if isLongExpression:
                expression += " " + np.random.choice(["that is", "which is"]) + " "
                if objectNode.locationComponent in self.expressionComponents.location.relativeLocationToMultipleObjects.allComponents:
                    if objectNode.locationComponent == self.expressionComponents.location.relativeLocationToMultipleObjects.middle:
                        expression += (
                            "in the middle of the "
                            + np.random.choice(["", objectNode.children[0].color + " "])
                            + np.random.choice([objectNode.children[0].name, objectNode.children[0].category])
                            + " and the "
                            + np.random.choice(["", objectNode.children[1].color + " "])
                            + np.random.choice([objectNode.children[1].name, objectNode.children[1].category])
                        )
                    elif objectNode.locationComponent == self.expressionComponents.location.relativeLocationToMultipleObjects.rowOrder:
                        orderWords = ["first", "second", "third", "fourth"]
                        leftOrRightWords = ["left", "right"]
                        leftOrRightWord = np.random.choice(leftOrRightWords)
                        if leftOrRightWord == "left":
                            expression += orderWords[objectNode.order - 1] + " from left"
                        else:
                            expression += orderWords[4 - objectNode.order] + " from right"
                elif objectNode.locationComponent in self.expressionComponents.location.absoluteLocation.allComponents:
                    absoluteLocationWords = GetAbsoluteLocationWords(objectNode.locationComponent)
                    expression += np.random.choice(absoluteLocationWords)
                elif objectNode.locationComponent == self.expressionComponents.location.randomLocation:
                    if objectNode.attributeComponent == self.expressionComponents.attribute.indirectRefer.notColor:
                        expression += "not " + objectNode.children[0].color
                    elif (
                        objectNode.attributeComponent == self.expressionComponents.attribute.randomAttribute
                        or objectNode.attributeComponent == self.expressionComponents.attribute.directRefer.color
                    ):
                        expression += np.random.choice(["in", ""]) + " " + objectNode.color
                    else:
                        raise SceneGraphGenerationError(
                            f"as a parent, locationComponent is random location, but attributeComponent is not valid:\n{objectNode}\n"
                            + f"scene graph:\n{self}"
                        )
                else:
                    raise SceneGraphGenerationError(
                        f"locationComponent is not a valid location component while generating text with this "
                        + f"objectNode as parent:\n{objectNode}\n"
                        + f"scene graph:\n{self}"
                    )
            return expression, isLongExpression

        expression = "the "
        if objectNode.attributeComponent == self.expressionComponents.attribute.directRefer.color:
            expression += objectNode.color + " "
        if objectNode.objectReferComponent == self.expressionComponents.objectRefer.name:
            expression += objectNode.name + " "
        elif objectNode.objectReferComponent == self.expressionComponents.objectRefer.category:
            expression += objectNode.category + " "
        elif objectNode.objectReferComponent == self.expressionComponents.objectRefer.ambiguous:
            expression += np.random.choice(["stuff", "thing", "object"]) + " "
        else:
            raise SceneGraphGenerationError(f"objectReferComponent is not a valid object refer component:\n{objectNode}\nscne graph:\n{self}")

        # check location component
        clause_words = ["that is ", "which is ", ""]
        if objectNode.locationComponent in self.expressionComponents.location.relativeLocationToSingleObject.allComponents:
            if objectNode.locationComponent == self.expressionComponents.location.relativeLocationToSingleObject.front:
                location_phrases = ["in front of", "in the front of", "ahead of"]
            elif objectNode.locationComponent == self.expressionComponents.location.relativeLocationToSingleObject.rear:
                location_phrases = ["behind", "in the back of", "at the back of"]
            elif objectNode.locationComponent == self.expressionComponents.location.relativeLocationToSingleObject.left:
                location_phrases = ["on the left of", "to the left of", "left of"]
            elif objectNode.locationComponent == self.expressionComponents.location.relativeLocationToSingleObject.right:
                location_phrases = ["on the right of", "to the right of", "right of"]
            elif objectNode.locationComponent == self.expressionComponents.location.relativeLocationToSingleObject.frontLeft:
                location_phrases = [
                    "in front of and to the left of",
                    "in the front left of",
                    "at the front left of",
                    "front left of",
                    "to the left of and in front of",
                    "to the left front of",
                    "to the left and front of",
                    "left front of",
                ]
            elif objectNode.locationComponent == self.expressionComponents.location.relativeLocationToSingleObject.frontRight:
                location_phrases = [
                    "in front of and to the right of",
                    "in the front right of",
                    "at the front right of",
                    "front right of",
                    "to the right of and in front of",
                    "to the right front of",
                    "to the right and front of",
                    "right front of",
                ]
            elif objectNode.locationComponent == self.expressionComponents.location.relativeLocationToSingleObject.rearLeft:
                location_phrases = [
                    "behind and to the left of",
                    "in the rear left of",
                    "at the rear left of",
                    "rear left of",
                    "to the left of and behind",
                    "to the left rear of",
                    "to the left and rear of",
                    "left rear of",
                ]
            elif objectNode.locationComponent == self.expressionComponents.location.relativeLocationToSingleObject.rearRight:
                location_phrases = [
                    "behind and to the right of",
                    "in the rear right of",
                    "at the rear right of",
                    "rear right of",
                    "to the right of and behind",
                    "to the right rear of",
                    "to the right and rear of",
                    "right rear of",
                ]
            elif objectNode.locationComponent == self.expressionComponents.location.relativeLocationToSingleObject.near:
                location_phrases = ["near", "close to", "next to", "beside", "by"]

            parent_expression, is_long_expression = self.GetExpression(objectNode.parent, is_parent=True)
            if is_long_expression:
                expression += np.random.choice(location_phrases) + " " + parent_expression
            else:
                expression += np.random.choice(clause_words) + np.random.choice(location_phrases) + " " + parent_expression
        elif objectNode.locationComponent in self.expressionComponents.location.relativeLocationToMultipleObjects.allComponents:
            if objectNode.locationComponent == self.expressionComponents.location.relativeLocationToMultipleObjects.middle:
                expression += (
                    "in the middle of the "
                    + np.random.choice(["", objectNode.children[0].color + " "])
                    + np.random.choice([objectNode.children[0].name, objectNode.children[0].category])
                    + " and the "
                    + np.random.choice(["", objectNode.children[1].color + " "])
                    + np.random.choice([objectNode.children[1].name, objectNode.children[1].category])
                )
            elif objectNode.locationComponent == self.expressionComponents.location.relativeLocationToMultipleObjects.rowOrder:
                orderWords = ["first", "second", "third", "fourth"]
                leftOrRightWords = ["left", "right"]
                leftOrRightWord = np.random.choice(leftOrRightWords)
                if leftOrRightWord == "left":
                    expression += np.random.choice(clause_words) + orderWords[objectNode.order - 1] + " from left"
                else:
                    expression += orderWords[4 - objectNode.order] + " from right"
        elif objectNode.locationComponent in self.expressionComponents.location.absoluteLocation.allComponents:
            absoluteLocationWords = GetAbsoluteLocationWords(objectNode.locationComponent)
            expression += np.random.choice(clause_words) + np.random.choice(absoluteLocationWords)
        elif objectNode.locationComponent == self.expressionComponents.location.randomLocation:
            # check attribute component
            if objectNode.attributeComponent == self.expressionComponents.attribute.indirectRefer.sameColor:
                phrases = ["that has the same color as", "which has the same color as", "whose color is the same as"]
                expression += np.random.choice(phrases) + " " + self.GetExpression(objectNode.parent, is_parent=True, get_parent_color=False)[0]
            elif objectNode.attributeComponent == self.expressionComponents.attribute.indirectRefer.notColor:
                phrases = ["that is not", "which is not", "whose color is not"]
                expression += np.random.choice(phrases) + " " + objectNode.children[0].color
            elif objectNode.attributeComponent == self.expressionComponents.attribute.randomAttribute:
                pass
            elif objectNode.attributeComponent == self.expressionComponents.attribute.directRefer.color:
                pass
            else:
                raise SceneGraphGenerationError(f"attributeComponent is not a valid attribute component:\n{objectNode}\nscne graph:\n{self}")
        else:
            raise SceneGraphGenerationError(f"locationComponent is not a valid location component:\n{objectNode}\nscne graph:\n{self}")
        return expression

    def GetSimpleReferringExpressions(self) -> list[str]:
        """
        Get a list of simple referring expressions for each object node with expression components in the scene graph.
        Object nodes come from `self.referringExpressionStructures.keys()`.

        Returns:
            list[str]: A list of simple referring expressions for each object node with expression components in the scene graph.
        """
        # TODO: color attribute only for now
        expressions = []
        for objectNode in self.referringExpressionStructures.keys():
            expressions.append(self.GetExpression(objectNode))
        return expressions

    def GetComplexReferringExpressions(self) -> list[str]:
        actions = ["grab", "pick up", "take", "get", "hold", "grasp", ""]
        actions_without_ending = ["grab me", "get me"]
        requests = ["can you", "could you", "would you", "will you"]
        orders = ["I'd like to", "I want to", "I would like to"]
        questions = ["where is", "what is", "which is", "what's", "where's", "which one is"]
        direct_starters = ["I want", "I need", "I'd like", "I would like"]
        orders2 = ["I want you to", "I would like you to", "I'd like you to", "I need you to"]
        endings = ["for me", ""]
        pleases = ["please", ""]
        # case 0: actions + simple expression + endings: (grab) the red apple (for me)
        # case 1: action_without_ending + simple expression: grab me the red apple
        # case 2: requests + actions[:-1] + simple expression + endings: can you grab the red apple (for me)
        # case 3: requests + actions_without_ending + simple expression: can you grab me the red apple
        # case 4: orders + actions[:-1] + simple expression: I'd like to grab the red apple
        # case 5: questions + simple expression: where is the red apple
        # case 6: direct_starters + simple expression : I want the red apple
        # case 7: orders2 + actions[:-1] + simple expression + endings: I want you to grab the red apple (for me)
        # case 8: orders2 + actions_without_ending + simple expression: I want you to grab me the red apple

        def MakeSentenceFromList(expressionList: list[str]) -> str:
            sentence = ""
            for item in expressionList:
                sentence += item + " "
            # remove extra spaces
            sentence = re.sub(" +", " ", sentence).strip()
            sentence = re.sub(r"\s([?.!,](?:\s|$))", r"\1", sentence)
            # upper the first letter
            sentence = sentence[0].upper() + sentence[1:]
            return sentence

        complexExpressions = []
        for simpleExpression in self.GetSimpleReferringExpressions():
            caseNum = np.random.choice(7)
            match caseNum:
                case 0:
                    action = np.random.choice(actions)
                    if action != "":
                        ending = np.random.choice(endings)
                    else:
                        ending = ""
                    expressionList = [action, simpleExpression, ending, "."]
                    if action != "":
                        idxsToInsertPleases = [0, 3]
                    else:
                        idxsToInsertPleases = [3]
                case 1:
                    expressionList = [np.random.choice(actions_without_ending), simpleExpression, "."]
                    idxsToInsertPleases = [0, 2]
                case 2:
                    expressionList = [np.random.choice(requests), np.random.choice(actions[:-1]), simpleExpression, np.random.choice(endings), "?"]
                    idxsToInsertPleases = [0, 1, 4]
                case 3:
                    expressionList = [np.random.choice(requests), np.random.choice(actions_without_ending), simpleExpression, "?"]
                    idxsToInsertPleases = [1, 3]
                case 4:
                    expressionList = [np.random.choice(orders), np.random.choice(actions[:-1]), simpleExpression, "."]
                    idxsToInsertPleases = [3]
                case 5:
                    expressionList = [np.random.choice(questions), simpleExpression, "?"]
                    idxsToInsertPleases = []
                case 6:
                    expressionList = [np.random.choice(direct_starters), simpleExpression, "."]
                    idxsToInsertPleases = [2]
                case 7:
                    expressionList = [np.random.choice(orders2), np.random.choice(actions[:-1]), simpleExpression, np.random.choice(endings), "."]
                    idxsToInsertPleases = [4]
                case 8:
                    expressionList = [np.random.choice(orders2), np.random.choice(actions_without_ending), simpleExpression, "."]
                    idxsToInsertPleases = [3]
            if idxsToInsertPleases != []:
                expressionList.insert(np.random.choice(idxsToInsertPleases), np.random.choice(pleases))
            complexExpressions.append(MakeSentenceFromList(expressionList))
        return complexExpressions

    def CheckForbiddenExpressions(
        self, obj: list, locationComponent: int, attributeComponent: int, objectReferComponent: int, coordinate: tuple
    ) -> bool:
        expressionComponentsForCheck = [
            {self.expressionComponents.attribute.directRefer.color: obj[3], self.expressionComponents.objectRefer.name: obj[1]},
            {self.expressionComponents.attribute.directRefer.color: obj[3], self.expressionComponents.objectRefer.category: obj[2]},
            {self.expressionComponents.attribute.directRefer.color: obj[3], self.expressionComponents.objectRefer.ambiguous: True},
            {self.expressionComponents.attribute.randomAttribute: True, self.expressionComponents.objectRefer.name: obj[1]},
            {self.expressionComponents.attribute.randomAttribute: True, self.expressionComponents.objectRefer.category: obj[2]},
            {self.expressionComponents.attribute.randomAttribute: True, self.expressionComponents.objectRefer.ambiguous: True},
        ]
        if (
            locationComponent == self.expressionComponents.location.randomLocation
            or locationComponent == self.expressionComponents.location.relativeLocationToMultipleObjects.rowOrder
            or attributeComponent in self.expressionComponents.attribute.indirectRefer.allComponents
        ):
            for item in expressionComponentsForCheck:
                for forbiddenExpressionComponentsRegion in self.forbiddenExpressionComponentsGlobal:
                    if Util.DictIn(item, forbiddenExpressionComponentsRegion):
                        return False

        for item in expressionComponentsForCheck:
            for forbiddenExpressionComponentsRegion in self.forbiddenExpressionComponentsRegionMap[coordinate]:
                if Util.DictIn(item, forbiddenExpressionComponentsRegion):
                    return False
        return True

    def __str__(self) -> str:
        string = "Map:\n"
        for i in range(self.mapSize[0]):
            for j in range(self.mapSize[1]):
                string += f"{(self.map[i, j].name if self.map[i, j] is not None else '.'):^30}"
            string += "\n"
        string += "\nObjectNodes:\n"
        for objectNode in self.objectNodes:
            string += f"{objectNode}\n"
        return string

    def SaveScene(self, fileName: str):
        """
        Save the scene graph to a file.\n

        Args:
            fileName (str): The file name of the scene graph.
        """
        with open(fileName, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def WriteSceneGraphToFile(sceneGraph: SceneGraph, save_path: str, file_name_prefix: str) -> int:
        """
        Save the scene graph to a folder that may contain multiple scene graphs.\n
        The file is named like "`file_name_prefix`_00000012.pkl".

        Args:
            sceneGraph (SceneGraph): The scene graph.
            save_path (str): The folder path.
            name_suffix (str): The file name suffix.

        Returns:
            int: The index of created scene graph file.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        num_existing_files = len(os.listdir(save_path))
        file_name = f"{file_name_prefix}_{num_existing_files:08d}.pkl"
        sceneGraph.SaveScene(os.path.join(save_path, file_name))
        return num_existing_files

    @staticmethod
    def LoadScene(fileName: str) -> SceneGraph:
        """
        Load a scene graph from a file.\n

        Args:
            fileName (str): The file name of the scene graph.

        Returns:
            SceneGraph: The scene graph.
        """
        with open(fileName, "rb") as f:
            sceneGraph = pickle.load(f)
        return sceneGraph


class ObjectNode:
    """
    [WIP] An object node in the scene graph.\n

    """

    def __init__(
        self,
        scene: SceneGraph,
        obj: str,
        name: str,
        category: str,
        coordinate: tuple[int, int],
        color: str,
        size: float,
        referringExpressionStructure: list,
        parent: ObjectNode = None,
        order: int = -1,
    ) -> None:
        """
        Create an object node in the scene graph.\n

        Args:
            scene (SceneGraph): The scene graph that the object node belongs to.
            obj (str): The obj file name of the object.
            name (str): The name of the object.
            category (str): The category of the object.
            coordinate (tuple[int, int]): The coordinate of the object in the scene map.
            color (str): The color of the object.
            size (float): The size of the object.
            referringExpressionStructure (list): A list of `[location, attribute, objectRefer]` components.
            parent (ObjectNode, optional): Parent object node. Defaults to None.
            order (int, optional): The order if `location` is `rowOrder`. Defaults to -1.
        """
        self.scene = scene
        self.scene_id = len(self.scene.objectNodes)
        self.obj_id = obj
        self.name = name
        self.category = category
        self.coordinate = coordinate
        self.color = color
        self.size = size
        self.order = order
        self.parent = parent
        self.children: list[ObjectNode] = []
        self.referringExpressionStructure = referringExpressionStructure
        if referringExpressionStructure != []:
            self.locationComponent = referringExpressionStructure[0]
            self.attributeComponent = referringExpressionStructure[1]
            self.objectReferComponent = referringExpressionStructure[2]
        else:
            self.locationComponent = None
            self.attributeComponent = None
            self.objectReferComponent = None
        self.Register()
        self.Generate()

    def Generate(self):
        # TODO: generate in blender
        pass

    def Register(self):
        self.scene.objectNodes.append(self)
        self.scene.map[self.coordinate] = self
        if self.referringExpressionStructure != []:
            self.scene.referringExpressionStructures[self] = [self.locationComponent, self.attributeComponent, self.objectReferComponent]
        self.UpdateForbiddenExpressionComponents()

    def UpdateForbiddenExpressionComponents(self):
        """
        [WIP] Update the forbidden expression components map of the scene graph.\n
        For each coordinate in the forbidden expression components map, there is a list of dictionaries of forbidden expression components.
        The dictionaries contain the expression component and value pair, e.g. `{location: "front", color: "red", objectRefer: "can"}`.
        Before a new object node is created at a coordinate, the forbidden expression components at this coordinate should be checked.
        """
        if self.referringExpressionStructure == []:
            return
        # TODO: attribute dict if more attribute components are added
        objectReferDict = {
            self.scene.expressionComponents.objectRefer.name: self.name,
            self.scene.expressionComponents.objectRefer.category: self.category,
            self.scene.expressionComponents.objectRefer.ambiguous: True,
        }

        targetCoordinates = self.scene.GetLocationCoordinatesFromLocationComponent(self.locationComponent, self.parent, self.order)
        # TODO revise if more attribute
        if (
            self.attributeComponent == self.scene.expressionComponents.attribute.directRefer.color
            or self.attributeComponent == self.scene.expressionComponents.attribute.indirectRefer.sameColor
        ):
            components = {
                self.scene.expressionComponents.attribute.directRefer.color: self.color,
                self.objectReferComponent: objectReferDict[self.objectReferComponent],
            }
        elif (
            self.attributeComponent == self.scene.expressionComponents.attribute.randomAttribute
            or self.attributeComponent == self.scene.expressionComponents.attribute.indirectRefer.notColor
        ):
            components = {
                self.scene.expressionComponents.attribute.randomAttribute: True,
                self.objectReferComponent: objectReferDict[self.objectReferComponent],
            }
        if components not in self.scene.forbiddenExpressionComponentsGlobal:
            self.scene.forbiddenExpressionComponentsGlobal.append(components)

        if (
            self.locationComponent in self.scene.expressionComponents.location.relativeLocationToSingleObject.allComponents
            or self.locationComponent in self.scene.expressionComponents.location.absoluteLocation.allComponents
        ):
            for coordinate in targetCoordinates:
                if Util.DictIn(components, self.scene.forbiddenExpressionComponentsRegionMap[coordinate]) is False:
                    self.scene.forbiddenExpressionComponentsRegionMap[coordinate].append(components)

        if (
            self.locationComponent == self.scene.expressionComponents.location.relativeLocationToMultipleObjects.rowOrder
            or self.locationComponent == self.scene.expressionComponents.location.randomLocation
            or self.attributeComponent in self.scene.expressionComponents.attribute.indirectRefer.allComponents
        ):
            targetCoordinates = self.scene.GetLocationCoordinatesFromLocationComponent(
                self.scene.expressionComponents.location.randomLocation, self.parent, self.order, filter=True
            )
            for coordinate in targetCoordinates:
                if Util.DictIn(components, self.scene.forbiddenExpressionComponentsRegionMap[coordinate]) is False:
                    self.scene.forbiddenExpressionComponentsRegionMap[coordinate].append(components)

    def AddChild(
        self, obj: str, name: str, category: str, coordinate: tuple[int, int], color: str, size: float, referringExpressionStructure: list
    ) -> ObjectNode:
        """
        Add a child object node to the object node.\n

        Args:
            obj (str): _description_
            name (str): _description_
            category (str): _description_
            coordinate (tuple[int, int]): _description_
            color (str): _description_
            size (float): _description_
            referringExpressionStructure (list): _description_

        Returns:
            ObjectNode: _description_
        """
        child = ObjectNode(self.scene, obj, name, category, coordinate, color, size, referringExpressionStructure, parent=self)
        self.children.append(child)
        return child

    def SatisfyExpression(self, expression: dict) -> bool:
        objectReferDict = {
            self.scene.expressionComponents.objectRefer.name: self.name,
            self.scene.expressionComponents.objectRefer.category: self.category,
            self.scene.expressionComponents.objectRefer.ambiguous: True,
        }
        if (
            self.scene.expressionComponents.attribute.directRefer.color in expression
            and expression[self.scene.expressionComponents.attribute.directRefer.color] != self.color
        ):
            return False
        if (
            self.scene.expressionComponents.attribute.indirectRefer.notColor in expression
            and expression[self.scene.expressionComponents.attribute.indirectRefer.notColor] == self.color
        ):
            return False

        for objectRefer in objectReferDict.keys():
            if objectRefer in expression and expression[objectRefer] != objectReferDict[objectRefer]:
                return False
        return True

    def __str__(self) -> str:
        return (
            f"ObjectNode:{self.name}(scene_id={self.scene_id}, obj_id={self.obj_id}, coordinate={self.coordinate}, color={self.color}, size={self.size},\n"
            + f"           parent={f'{self.parent.name} at {self.parent.coordinate}' if self.parent is not None else None},\n"
            + f"           children={[f'{child.name} at {child.coordinate}' for child in self.children]},\n"
            + f"           locationComponent={self.scene.expressionComponents.GetComponentName(self.locationComponent)},\n"
            + f"           attributeComponent={self.scene.expressionComponents.GetComponentName(self.attributeComponent)},\n"
            + f"           objectReferComponent={self.scene.expressionComponents.GetComponentName(self.objectReferComponent)})\n"
        )

    def __repr__(self) -> str:
        return self.__str__()


class _ExpressionComponentsBase:
    @functools.cached_property
    def allComponents(self) -> list[int]:
        """
        Get a list of all components in the specified expression components class.

        Returns:
            list[int]: A list of all components in the specified expression components class.
        """
        result = []
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, (int, str, bool, float)):
                result.append(attr_value)
            elif hasattr(attr_value, "__dict__"):
                result.extend(attr_value.allComponents)
        return result

    def GetRandomComponent(self) -> int:
        """
        Get a random component in the specified expression components class.

        Returns:
            int: A random component in the specified expression components class.
        """
        return np.random.choice(self.allComponents)

    @functools.cache
    def GetComponentName(self, componentId: int) -> str:
        """
        Get the name of the specified component for a given component id.

        Args:
            componentId (int): The component id.

        Returns:
            str: The component name, None if the component id is not found.
        """
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, (int, str, bool, float)):
                if attr_value == componentId:
                    return attr_name
            elif hasattr(attr_value, "__dict__"):
                result = attr_value.GetComponentName(componentId)
                if result is not None:
                    return f"{attr_name}.{result}"
        return None


class ExpressionComponents(_ExpressionComponentsBase):
    """
    Assign an integer to each expression component, can be perceived as a struct.\n
    e.g. `expressionComponents.location.relativeLocationToSingleObject.front` is 0.\n
    `expressionComponents.objectRefer.name` is 700.\n

    Args:
        _ExpressionComponentsBase (_type_): _description_
    """

    def __init__(self) -> None:
        self.location = self._Location()
        self.attribute = self._Attribute()
        self.objectRefer = self._ObjectRefer()

    class _Location(_ExpressionComponentsBase):
        """0-399"""

        def __init__(self) -> None:
            self.relativeLocationToSingleObject = self._RelativeLocationToSingleObject()
            self.relativeLocationToMultipleObjects = self._RelativeLocationToMultipleObjects()
            self.absoluteLocation = self._AbsoluteLocation()
            self.randomLocation = 300

        class _RelativeLocationToSingleObject(_ExpressionComponentsBase):
            def __init__(self) -> None:
                self.front = 0
                self.rear = 1
                self.left = 2
                self.right = 3
                self.frontLeft = 4
                self.frontRight = 5
                self.rearLeft = 6
                self.rearRight = 7
                self.near = 8
                # TODO: top bottom not

        class _RelativeLocationToMultipleObjects(_ExpressionComponentsBase):
            def __init__(self) -> None:
                self.middle = 100
                self.rowOrder = 101
                # TODO: columnOrder front rear left right...

        class _AbsoluteLocation(_ExpressionComponentsBase):
            def __init__(self) -> None:
                self.frontmost = 200
                self.rearmost = 201
                self.leftmost = 202
                self.rightmost = 203
                self.frontLeftmost = 204
                self.frontRightmost = 205
                self.rearLeftmost = 206
                self.rearRightmost = 207

    class _Attribute(_ExpressionComponentsBase):
        """400-699"""

        def __init__(self) -> None:
            self.directRefer = self._DirectRefer()
            self.indirectRefer = self._IndirectRefer()
            self.randomAttribute = 600

        class _DirectRefer(_ExpressionComponentsBase):
            def __init__(self) -> None:
                self.color = 400
                # self.size = 401
                # TODO shape material

        class _IndirectRefer(_ExpressionComponentsBase):
            def __init__(self) -> None:
                self.notColor = 500
                self.sameColor = 501
                # TODO shape material
                pass

    class _ObjectRefer(_ExpressionComponentsBase):
        def __init__(self) -> None:
            self.name = 700
            self.category = 701
            self.ambiguous = 702


class SceneGraphGenerationError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.args[0]}"


class Util:
    @staticmethod
    def DictIn(a: dict, b: dict) -> bool:
        for key, value in a.items():
            if key not in b or b[key] != value:
                return False
        return True
