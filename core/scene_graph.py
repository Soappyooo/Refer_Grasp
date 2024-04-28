from __future__ import annotations
import numpy as np
import pandas as pd
import pickle
import os
import re
from utils.misc import Misc
from core.object_node import ObjectNode
from core.expression_components import ExpressionComponents


class SceneGraph:
    """
    [WIP] A scene graph with object nodes, map and expressions.
    """

    def __init__(self, map_size: tuple[int, int]) -> None:
        """
        Create an empty scene with a map of size `map_size`.

        Args:
            map_size (tuple[int, int]): The size of the scene map. e.g. (4, 4) for a 4x4 map with 16 cells for placing objects.
        """
        self.components = ExpressionComponents()
        self.object_nodes: list[ObjectNode] = []
        self.map_size = map_size
        self.map = np.empty(map_size, dtype=ObjectNode)
        # each cell contains a list of forbidden expression components
        self.banned_components_regional = np.empty(map_size, dtype=object)
        self.banned_components_global: list[dict] = []
        for i in range(map_size[0]):
            for j in range(map_size[1]):
                self.banned_components_regional[i, j] = []
        self.expression_structures: dict[ObjectNode, list] = {}
        self.dataset: list = []
        self.retry_times = 50
        self.generated_expressions: list[str] = None

    def create_scene(self, min_objects_number: int, max_retries: int = None) -> bool:
        """
        Create a scene with at least `min_objects_number` objects. Objects information should be loaded before creating the scene.

        Args:
            min_objects_number (int): The minimum number of objects in the scene.
            max_retries (int, optional): The maximum number of retries. Defaults to None.

        Returns:
            bool: True if the scene is created successfully, False otherwise.
        """
        if self.dataset == []:
            raise RuntimeError("No objects information loaded")
        self.clear_scene()
        if max_retries is not None:
            retry = max_retries
        else:
            retry = self.retry_times
        while retry:
            self.create_tree()
            if len(self.object_nodes) >= min_objects_number:
                return True
            retry -= 1
        return False

    def clear_scene(self):
        """
        Reset the scene graph to an empty scene.
        """
        self.object_nodes.clear()
        self.map = np.empty(self.map_size, dtype=ObjectNode)
        self.banned_components_regional = np.empty(self.map_size, dtype=list)
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                self.banned_components_regional[i, j] = []
        self.expression_structures.clear()
        self.banned_components_global.clear()

    def add_base_object_node(
        self,
        obj: str,
        name: str,
        category: str,
        coordinate: tuple[int, int],
        color: str,
        size: float,
        referring_expression_structure: list,
        order: int = -1,
        aliases: list = None,
    ) -> ObjectNode:
        baseObjectNode = ObjectNode(self, obj, name, category, coordinate, color, size, referring_expression_structure, order=order, aliases=aliases)
        return baseObjectNode

    def get_random_components(self) -> list[int]:
        """
        Get a list of random `[location, attribute, refer]` components.

        Returns:
            list[int]: A list of random `[location, attribute, refer]` components.
        """
        return [
            self.components.location.get_random_component(),
            self.components.attribute.get_random_component(),
            self.components.refer.get_random_component(),
        ]

    def get_base_object_components(self) -> list[int]:
        """
        Get a list of random `[location, attribute, refer]` components for base object node.
        Location can be relative to multiple, absolute, or unspecified. Attribute can be direct, not_color, or unspecified. Refer can be any.
        Note that `[unspecified, unspecified, ambiguous]` is not allowed.

        Returns:
            list[int]: A list of random `[location, attribute, refer]` components.
        """
        while True:
            components = [
                ExpressionComponents.randomly_select_from(
                    self.components.location.relative_to_multiple.all_components,
                    self.components.location.absolute.all_components,
                    self.components.location.unspecified,
                ),
                ExpressionComponents.randomly_select_from(
                    self.components.attribute.direct.all_components,
                    self.components.attribute.indirect.not_color,
                    self.components.attribute.unspecified,
                ),
                ExpressionComponents.randomly_select_from(self.components.refer.all_components),
            ]
            if components[1] == self.components.attribute.indirect.not_color:
                components[0] = self.components.location.unspecified
                if components[2] == self.components.refer.ambiguous:
                    continue
            if components != [self.components.location.unspecified, self.components.attribute.unspecified, self.components.refer.ambiguous]:
                break

        return components

    def load_objects_info(self, file_path: str):
        """
        Load objects information from a supported file. File format should be `.xlsx` for now.\n

        Args:
            file_path (str): The file path of the dataset.

        Raises:
            RuntimeError: If the file format is not supported.
        """
        if file_path.endswith(".xlsx"):
            self.dataset = pd.read_excel(file_path, keep_default_na=False).values.tolist()
        else:
            raise RuntimeError("Dataset file format not supported")

    def create_tree(self) -> bool:
        """
        [WIP] Create a tree structure with a base object node and some child object nodes.\n
        The base object node should not have relative location to single object or indirect attribute since it has no parent.
        If the base object node has relative location to multiple objects, the objects related to the base object node are generated as its children.
        If the base object node has absolute location or random location, the base object node is generated at a possible location.\n
        If the base object node has attribute 'notColor', the child will have the specified color.\n
        Then, a child object node with relative location to single object (the base object node) is generated. \n
        Will consider forbidden expression components later.
        """

        retry = self.retry_times
        while retry:
            base_object_components = self.get_base_object_components()
            order = np.random.randint(1, 5)
            possibleCoordinatesList = self.get_coordinates(base_object_components[0], order=order)
            if possibleCoordinatesList == []:
                retry -= 1
            else:
                break
        if retry == 0:
            return False

        retry = self.retry_times
        while True:
            retryFlag = False

            retry -= 1
            if retry == 0:
                return False

            if base_object_components[0] in self.components.location.relative_to_multiple.all_components:
                coordinates = possibleCoordinatesList[np.random.choice(len(possibleCoordinatesList))]
                # if middle, randomly choose 3 objects
                if base_object_components[0] == self.components.location.relative_to_multiple.middle:
                    baseObj = self.dataset[np.random.choice(len(self.dataset))]
                    # consider object with no color
                    if baseObj[3] == "":
                        base_object_components[1] = self.components.attribute.unspecified
                    if (
                        self.check_forbidden_expressions(
                            baseObj, base_object_components[0], base_object_components[1], base_object_components[2], coordinates[0]
                        )
                        is False
                    ):
                        retryFlag = True
                        continue
                    else:
                        leftObj = self.dataset[np.random.choice(len(self.dataset))]
                    if self.check_forbidden_expressions(leftObj, None, None, None, coordinates[1]) is False:
                        retryFlag = True
                        continue
                    else:
                        rightObj = self.dataset[np.random.choice(len(self.dataset))]
                    if self.check_forbidden_expressions(rightObj, None, None, None, coordinates[2]) is False:
                        retryFlag = True
                        continue
                    else:
                        baseObjectNode = self.add_base_object_node(
                            baseObj[0],
                            baseObj[1],
                            baseObj[2],
                            coordinates[0],
                            baseObj[3],
                            1,
                            base_object_components,
                            aliases=[] if baseObj[4] == "" else baseObj[4].split(";"),
                        )
                        baseObjectNode.AddChild(
                            leftObj[0],
                            leftObj[1],
                            leftObj[2],
                            coordinates[1],
                            leftObj[3],
                            1,
                            [],
                            aliases=[] if leftObj[4] == "" else leftObj[4].split(";"),
                        )
                        baseObjectNode.AddChild(
                            rightObj[0],
                            rightObj[1],
                            rightObj[2],
                            coordinates[2],
                            rightObj[3],
                            1,
                            [],
                            aliases=[] if rightObj[4] == "" else rightObj[4].split(";"),
                        )
                # if row order, randomly choose 4 same objects
                elif base_object_components[0] == self.components.location.relative_to_multiple.row_order:
                    baseObj = self.dataset[np.random.choice(len(self.dataset))]
                    if baseObj[3] == "":
                        base_object_components[1] = self.components.attribute.unspecified
                    base_object_components[2] = np.random.choice([self.components.refer.name, self.components.refer.category])
                    for i in range(1, len(coordinates)):
                        if (
                            self.check_forbidden_expressions(
                                baseObj, base_object_components[0], base_object_components[1], base_object_components[2], coordinates[i]
                            )
                            is False
                        ):
                            retryFlag = True
                            break
                    if retryFlag is False:
                        baseObjectNode = self.add_base_object_node(
                            baseObj[0],
                            baseObj[1],
                            baseObj[2],
                            coordinates[0],
                            baseObj[3],
                            1,
                            base_object_components,
                            order=order,
                            aliases=[] if baseObj[4] == "" else baseObj[4].split(";"),
                        )
                        for i in range(1, len(coordinates)):
                            baseObjectNode.AddChild(
                                baseObj[0],
                                baseObj[1],
                                baseObj[2],
                                coordinates[i],
                                baseObj[3],
                                1,
                                [],
                                aliases=[] if baseObj[4] == "" else baseObj[4].split(";"),
                            )
                    else:
                        continue
            # if absolute or unspecified location, randomly choose an object
            elif (
                base_object_components[0] in self.components.location.absolute.all_components
                or base_object_components[0] == self.components.location.unspecified
            ) and base_object_components[1] != self.components.attribute.indirect.not_color:

                coordinate = possibleCoordinatesList[np.random.choice(len(possibleCoordinatesList))]
                baseObj = self.dataset[np.random.choice(len(self.dataset))]
                if baseObj[3] == "":
                    base_object_components[1] = self.components.attribute.unspecified
                # for absolute location, there should not be an object already satisfied the expression
                possibleCoordinatesToCheck = self.get_coordinates(base_object_components[0], None, filter=False)
                attributeDict = {
                    self.components.attribute.direct.color: baseObj[3],
                    self.components.attribute.unspecified: True,
                }
                objectReferDict = {
                    self.components.refer.name: baseObj[1],
                    self.components.refer.category: baseObj[2],
                    self.components.refer.ambiguous: True,
                }
                for coordinateToCheck in possibleCoordinatesToCheck:
                    if self.map[coordinateToCheck] is not None and self.map[coordinateToCheck].SatisfyExpression(
                        {
                            base_object_components[1]: attributeDict[base_object_components[1]],
                            base_object_components[2]: objectReferDict[base_object_components[2]],
                        }
                    ):
                        retryFlag = True
                        break

                if (
                    self.check_forbidden_expressions(
                        baseObj, base_object_components[0], base_object_components[1], base_object_components[2], coordinate
                    )
                    is False
                ):
                    retryFlag = True
                    continue
                if retryFlag is False:
                    baseObjectNode = self.add_base_object_node(
                        baseObj[0],
                        baseObj[1],
                        baseObj[2],
                        coordinate,
                        baseObj[3],
                        1,
                        base_object_components,
                        aliases=[] if baseObj[4] == "" else baseObj[4].split(";"),
                    )
                else:
                    continue
            # if notColor, create a random object with the specified color and one not specified color
            elif base_object_components[1] == self.components.attribute.indirect.not_color:
                childObj = self.dataset[np.random.choice(len(self.dataset))]
                if childObj[3] == "":
                    retryFlag = True
                    continue
                possibleBaseObjs = []
                for line in self.dataset:
                    if line[3] != childObj[3]:
                        if base_object_components[2] == self.components.refer.name and line[1] == childObj[1]:
                            possibleBaseObjs.append(line)
                        elif base_object_components[2] == self.components.refer.category and line[2] == childObj[2]:
                            possibleBaseObjs.append(line)
                if possibleBaseObjs == []:
                    retryFlag = True
                    continue

                baseObj = possibleBaseObjs[np.random.choice(len(possibleBaseObjs))]
                if baseObj[3] == "":
                    retryFlag = True
                    continue
                coordinate = possibleCoordinatesList[np.random.choice(len(possibleCoordinatesList))]
                coordinate2 = possibleCoordinatesList[np.random.choice(len(possibleCoordinatesList))]
                if (
                    self.check_forbidden_expressions(
                        baseObj, base_object_components[0], base_object_components[1], base_object_components[2], coordinate
                    )
                    is False
                    or self.check_forbidden_expressions(
                        childObj, base_object_components[0], base_object_components[1], base_object_components[2], coordinate2
                    )
                    is False
                ):
                    retryFlag = True
                    continue

                for objectNode in self.object_nodes:
                    if objectNode.SatisfyExpression({base_object_components[0]: baseObj[1], base_object_components[1]: childObj[3]}):
                        retryFlag = True
                        break

                if retryFlag is False:
                    baseObjectNode = self.add_base_object_node(
                        baseObj[0],
                        baseObj[1],
                        baseObj[2],
                        coordinate,
                        baseObj[3],
                        1,
                        base_object_components,
                        aliases=[] if baseObj[4] == "" else baseObj[4].split(";"),
                    )
                    baseObjectNode.AddChild(
                        childObj[0],
                        childObj[1],
                        childObj[2],
                        coordinate2,
                        childObj[3],
                        1,
                        [],
                        aliases=[] if childObj[4] == "" else childObj[4].split(";"),
                    )

            else:
                raise RuntimeError("baseObjctNodeComponents[0] is not a valid location component")

            if retryFlag is False:
                break

        # create child
        # TODO: child of order expression node
        if base_object_components[0] == self.components.location.relative_to_multiple.row_order:
            return True

        retry = self.retry_times
        while True:
            retry -= 1
            if retry == 0:
                return False
            childObjectNodeComponents = self.get_random_components()
            # only use relative location to single object for child object or same color
            if (
                childObjectNodeComponents[0] in self.components.location.relative_to_single.all_components
                and childObjectNodeComponents[1] not in self.components.attribute.indirect.all_components
            ):
                possibleCoordinatesList = self.get_coordinates(childObjectNodeComponents[0], baseObjectNode)
                if possibleCoordinatesList != []:
                    break
            elif childObjectNodeComponents[1] == self.components.attribute.indirect.same_color:
                childObjectNodeComponents[0] = self.components.location.unspecified
                possibleCoordinatesList = self.get_coordinates(childObjectNodeComponents[0], baseObjectNode)
                if possibleCoordinatesList != []:
                    break

        # for same color
        if childObjectNodeComponents[1] == self.components.attribute.indirect.same_color:
            possibleObjs = []
            for line in self.dataset:
                if line[3] == baseObjectNode.color and line[0] != baseObjectNode.obj_id and line[3] != "":
                    possibleObjs.append(line)
            if possibleObjs == []:
                return False

            retry = self.retry_times
            while True:
                childObj = possibleObjs[np.random.choice(len(possibleObjs))]
                coordinate = possibleCoordinatesList[np.random.choice(len(possibleCoordinatesList))]
                if (
                    self.check_forbidden_expressions(
                        childObj, childObjectNodeComponents[0], childObjectNodeComponents[1], childObjectNodeComponents[2], coordinate
                    )
                    is False
                ):
                    retryFlag = True

                if retryFlag is False:
                    baseObjectNode.AddChild(
                        childObj[0],
                        childObj[1],
                        childObj[2],
                        coordinate,
                        childObj[3],
                        1,
                        childObjectNodeComponents,
                        aliases=[] if childObj[4] == "" else childObj[4].split(";"),
                    )
                    return True
                else:
                    retry -= 1
                    if retry == 0:
                        return False

        # for direct color
        # TODO: check forbidden expression components
        retry = self.retry_times
        while True:

            retry -= 1
            if retry == 0:
                return False

            coordinate = possibleCoordinatesList[np.random.choice(len(possibleCoordinatesList))]
            childObj = self.dataset[np.random.choice(len(self.dataset))]
            if childObj[3] == "":
                childObjectNodeComponents[1] = self.components.attribute.unspecified
            retryFlag = False

            objectReferDict = {
                self.components.refer.name: childObj[1],
                self.components.refer.category: childObj[2],
                self.components.refer.ambiguous: True,
            }

            # for near location, the parent and child should not be the same object
            if childObjectNodeComponents[0] == self.components.location.relative_to_single.near:
                if childObj[0] == baseObj[0]:
                    retryFlag = True
                    continue

                # for near location, there should not be an object already satisfied the expression
                coordinatesAroundBaseObject = self.get_coordinates(childObjectNodeComponents[0], baseObjectNode, filter=False)
                for coordinateAroundBaseObject in coordinatesAroundBaseObject:
                    if self.map[coordinateAroundBaseObject] is not None and self.map[coordinateAroundBaseObject].SatisfyExpression(
                        {
                            self.components.attribute.direct.color: childObj[3],
                            childObjectNodeComponents[2]: objectReferDict[childObjectNodeComponents[2]],
                        }
                    ):
                        retryFlag = True
                        break

            if (
                self.check_forbidden_expressions(
                    childObj, childObjectNodeComponents[0], childObjectNodeComponents[1], childObjectNodeComponents[2], coordinate
                )
                is False
            ):
                retryFlag = True
                continue

            if retryFlag is False:
                break

        baseObjectNode.AddChild(
            childObj[0],
            childObj[1],
            childObj[2],
            coordinate,
            childObj[3],
            1,
            childObjectNodeComponents,
            aliases=[] if childObj[4] == "" else childObj[4].split(";"),
        )
        return True

    def get_coordinates(
        self, location_component: int, parent_object: ObjectNode = None, order: int = None, filter=True
    ) -> list[tuple[int, int]] | list[list[tuple[int, int]]]:
        """
        For a given `location_component`, return a list of coordinates that satisfy the `location_component` and filter the result with scene map.\n
        Specifically, if `location_component` is relative location to single object,
        return a list of coordinates that satisfy the `location_component` relative to `parent_object`.
        If `location_component` is relative location to multiple objects, return a 2D list of coordinates.
        For each row, the first coordinate indicates referred location,
        and the rest indicate the relavant objects' coordinates. Require `order` if `location_component` is `relative_to_multiple.row_order`.\n
        If `location_component` is absolute location, return a list with only one coordinate that satisfy the `location_component`.\n

        Args:
            location_component (int): An attribute of `self.expressionComponents.location`
            parent_object (ObjectNode, optional): The parent object node for `relative_to_single`. Defaults to None.
            order (int, optional): The order of the objects for `relative_to_multiple.row_order`. Defaults to None.
            filter (bool, optional): Whether to filter out coordinates which already have objects. Defaults to True.
        Raises:
            RuntimeError: If parameters are not valid.

        Returns:
            list[tuple[int, int]] or list[list[tuple[int, int]]]: for `relative_to_single` and `absolute`,
            return a list of coordinates; for `relative_to_multiple`, return a 2D list of coordinates.
            If no coordinate satisfies the `location_component`, return an empty list.
        """
        # TODO: revise if more location components are added
        coordinateList = []
        if location_component in self.components.location.relative_to_single.all_components and parent_object is None:
            raise RuntimeError("location_component is relative location to single object, but parentObjectNode is None")
        elif location_component == self.components.location.relative_to_multiple.row_order and order is None:
            raise RuntimeError("location_component is relative location to multiple objects (requires order), but order is None")

        if location_component in self.components.location.relative_to_single.all_components:
            location_mapping = {
                self.components.location.relative_to_single.front: (1, 0),
                self.components.location.relative_to_single.rear: (-1, 0),
                self.components.location.relative_to_single.left: (0, -1),
                self.components.location.relative_to_single.right: (0, 1),
                self.components.location.relative_to_single.front_left: (1, -1),
                self.components.location.relative_to_single.front_right: (1, 1),
                self.components.location.relative_to_single.rear_left: (-1, -1),
                self.components.location.relative_to_single.rear_right: (-1, 1),
            }

            # for front, rear, left, right, frontLeft, frontRight, rearLeft, rearRight
            for key, adjustment in location_mapping.items():
                if location_component == key:
                    coordinate = (parent_object.coordinate[0] + adjustment[0], parent_object.coordinate[1] + adjustment[1])
                    if self.is_coordinate_valid(coordinate):
                        coordinateList.append(coordinate)
            # for near
            if location_component == self.components.location.relative_to_single.near:
                for adjustment in location_mapping.values():
                    coordinate = (parent_object.coordinate[0] + adjustment[0], parent_object.coordinate[1] + adjustment[1])
                    if self.is_coordinate_valid(coordinate):
                        coordinateList.append(coordinate)

        elif location_component in self.components.location.relative_to_multiple.all_components:
            match location_component:
                case self.components.location.relative_to_multiple.middle:
                    for i in range(self.map_size[0]):
                        for j in range(1, self.map_size[1] - 1):
                            coordinates = [(i, j), (i, j - 1), (i, j + 1)]
                            coordinateList.append(coordinates)
                # TODO: consider 4 objects in a row for now
                case self.components.location.relative_to_multiple.row_order:
                    objectsNumberInRow = 4
                    if order <= 0 or order > objectsNumberInRow:
                        raise RuntimeError("order should be in range [1, objectsNumberInRow]")
                    for i in range(self.map_size[0]):
                        for j in range(order - 1, self.map_size[1] - objectsNumberInRow + order):
                            coordinates = [(i, j)]
                            for k in range(objectsNumberInRow):
                                coordinates.append((i, j + k - order + 1)) if k - order + 1 != 0 else None
                            coordinateList.append(coordinates)

        elif location_component in self.components.location.absolute.all_components:
            match location_component:
                case self.components.location.absolute.frontmost:
                    for i in range(self.map_size[1]):
                        coordinateList.append((self.map_size[0] - 1, i))
                case self.components.location.absolute.rearmost:
                    for i in range(self.map_size[1]):
                        coordinateList.append((0, i))
                case self.components.location.absolute.leftmost:
                    for i in range(self.map_size[0]):
                        coordinateList.append((i, 0))
                case self.components.location.absolute.rightmost:
                    for i in range(self.map_size[0]):
                        coordinateList.append((i, self.map_size[1] - 1))
                case self.components.location.absolute.front_leftmost:
                    coordinateList.append((self.map_size[0] - 1, 0))
                case self.components.location.absolute.front_rightmost:
                    coordinateList.append((self.map_size[0] - 1, self.map_size[1] - 1))
                case self.components.location.absolute.rear_leftmost:
                    coordinateList.append((0, 0))
                case self.components.location.absolute.rear_rightmost:
                    coordinateList.append((0, self.map_size[1] - 1))

        elif location_component == self.components.location.unspecified:
            for i in range(self.map_size[0]):
                for j in range(self.map_size[1]):
                    coordinateList.append((i, j))

        else:
            raise RuntimeError("location_component is not a valid location component")

        # filter out coordinates that are occupied
        filteredPossibleCoordinatesList = coordinateList.copy()
        if filter is True:
            if location_component in self.components.location.relative_to_multiple.all_components:
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

    def is_coordinate_valid(self, coordinate: tuple[int, int]) -> bool:
        """
        Check if the given `coordinate` is within the scene map.

        Args:
            coordinate (tuple[int, int]): The coordinate to check.

        Returns:
            bool: True if the `coordinate` is within the scene map, False otherwise.
        """
        return coordinate[0] >= 0 and coordinate[0] < self.map_size[0] and coordinate[1] >= 0 and coordinate[1] < self.map_size[1]

    def get_expression(self, object_node: ObjectNode, is_parent: bool = False, get_parent_color: bool = True) -> str | tuple[str, bool]:

        def get_absolute_location_words(location_component: int) -> str:
            if location_component == self.components.location.absolute.frontmost:
                absolute_location_words = [
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
            if location_component == self.components.location.absolute.rearmost:
                absolute_location_words = [
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
            if location_component == self.components.location.absolute.leftmost:
                absolute_location_words = [
                    "at the left",
                    "at left",
                    "leftmost",
                    "leftmost in the scene",
                    "in the left",
                    "in left",
                    "at the leftmost",
                    "in the leftmost of the scene",
                ]
            if location_component == self.components.location.absolute.rightmost:
                absolute_location_words = [
                    "at the right",
                    "at right",
                    "rightmost",
                    "rightmost in the scene",
                    "in the right",
                    "in right",
                    "at the rightmost",
                    "in the rightmost of the scene",
                ]
            if location_component == self.components.location.absolute.front_leftmost:
                absolute_location_words = [
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
                    "in left and front",
                    "in the left front corner",
                    "in left front corner",
                    "at the left front corner",
                    "in front and left",
                ]
            if location_component == self.components.location.absolute.front_rightmost:
                absolute_location_words = [
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
            if location_component == self.components.location.absolute.rear_leftmost:
                absolute_location_words = [
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
            if location_component == self.components.location.absolute.rear_rightmost:
                absolute_location_words = [
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
            return absolute_location_words

        if is_parent:
            expression = "the "
            colorRefer = np.random.choice([self.components.attribute.direct.color, self.components.attribute.unspecified])
            objectRefer = np.random.choice(
                [
                    self.components.refer.name,
                    self.components.refer.category,
                ]
            )
            referDict = {}
            if colorRefer == self.components.attribute.direct.color and get_parent_color:
                referDict[colorRefer] = object_node.color
                expression += object_node.color + " "
            if objectRefer == self.components.refer.name:
                referDict[objectRefer] = object_node.name
                expression += np.random.choice([object_node.name, *object_node.aliases])
            elif objectRefer == self.components.refer.category:
                referDict[objectRefer] = object_node.category
                expression += object_node.category

            isLongExpression = False
            for node in self.object_nodes:
                if node != object_node and node.SatisfyExpression(referDict):
                    isLongExpression = True
                    break
            if isLongExpression:
                expression += " " + np.random.choice(["that is", "which is"]) + " "
                if object_node.locationComponent in self.components.location.relative_to_multiple.all_components:
                    if object_node.locationComponent == self.components.location.relative_to_multiple.middle:
                        expression += (
                            (
                                "in the middle of the "
                                + np.random.choice(["", object_node.children[0].color + " "])
                                + np.random.choice([object_node.children[0].name, object_node.children[0].category, *object_node.children[0].aliases])
                                + " and the "
                                + np.random.choice(["", object_node.children[1].color + " "])
                                + np.random.choice([object_node.children[1].name, object_node.children[1].category, *object_node.children[1].aliases])
                            )
                            if object_node.children[0].obj_id != object_node.children[1].obj_id
                            else (
                                "in the middle of "
                                + np.random.choice(["the ", "two "])
                                + np.random.choice(["", object_node.children[0].color + " "])
                                + np.random.choice([object_node.children[0].name, object_node.children[0].category, *object_node.children[0].aliases])
                                + "s"
                            )
                        )
                    elif object_node.locationComponent == self.components.location.relative_to_multiple.row_order:
                        orderWords = ["first", "second", "third", "fourth"]
                        leftOrRightWords = ["left", "right"]
                        leftOrRightWord = np.random.choice(leftOrRightWords)
                        if leftOrRightWord == "left":
                            expression += orderWords[object_node.order - 1] + " from left"
                        else:
                            expression += orderWords[4 - object_node.order] + " from right"
                elif object_node.locationComponent in self.components.location.absolute.all_components:
                    absoluteLocationWords = get_absolute_location_words(object_node.locationComponent)
                    expression += np.random.choice(absoluteLocationWords)
                elif object_node.locationComponent == self.components.location.unspecified:
                    if object_node.attributeComponent == self.components.attribute.indirect.not_color:
                        expression += "not " + object_node.children[0].color
                    elif (
                        object_node.attributeComponent == self.components.attribute.unspecified
                        or object_node.attributeComponent == self.components.attribute.direct.color
                    ):
                        expression += np.random.choice(["in", ""]) + " " + object_node.color
                    else:
                        raise RuntimeError(
                            f"as a parent, locationComponent is random location, but attributeComponent is not valid:\n{object_node}\n"
                            + f"scene graph:\n{self}"
                        )
                else:
                    raise RuntimeError(
                        f"locationComponent is not a valid location component while generating text with this "
                        + f"objectNode as parent:\n{object_node}\n"
                        + f"scene graph:\n{self}"
                    )
            return expression, isLongExpression

        expression = "the "
        if object_node.attributeComponent == self.components.attribute.direct.color:
            expression += object_node.color + " "
        if object_node.objectReferComponent == self.components.refer.name:
            expression += np.random.choice([object_node.name, *object_node.aliases]) + " "
        elif object_node.objectReferComponent == self.components.refer.category:
            expression += object_node.category + " "
        elif object_node.objectReferComponent == self.components.refer.ambiguous:
            expression += np.random.choice(["stuff", "thing", "object"]) + " "
        else:
            raise RuntimeError(f"objectReferComponent is not a valid object refer component:\n{object_node}\nscne graph:\n{self}")

        # check location component
        clause_words = ["that is ", "which is ", ""]
        if object_node.locationComponent in self.components.location.relative_to_single.all_components:
            if object_node.locationComponent == self.components.location.relative_to_single.front:
                location_phrases = ["in front of", "in the front of", "ahead of"]
            elif object_node.locationComponent == self.components.location.relative_to_single.rear:
                location_phrases = ["behind", "in the back of", "at the back of"]
            elif object_node.locationComponent == self.components.location.relative_to_single.left:
                location_phrases = ["on the left of", "to the left of", "left of"]
            elif object_node.locationComponent == self.components.location.relative_to_single.right:
                location_phrases = ["on the right of", "to the right of", "right of"]
            elif object_node.locationComponent == self.components.location.relative_to_single.front_left:
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
            elif object_node.locationComponent == self.components.location.relative_to_single.front_right:
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
            elif object_node.locationComponent == self.components.location.relative_to_single.rear_left:
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
            elif object_node.locationComponent == self.components.location.relative_to_single.rear_right:
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
            elif object_node.locationComponent == self.components.location.relative_to_single.near:
                location_phrases = ["near", "close to", "next to", "beside", "by"]

            parent_expression, is_long_expression = self.get_expression(object_node.parent, is_parent=True)
            if is_long_expression:
                expression += np.random.choice(location_phrases) + " " + parent_expression
            else:
                expression += np.random.choice(clause_words) + np.random.choice(location_phrases) + " " + parent_expression
        elif object_node.locationComponent in self.components.location.relative_to_multiple.all_components:
            if object_node.locationComponent == self.components.location.relative_to_multiple.middle:
                expression += (
                    (
                        "in the middle of the "
                        + np.random.choice(["", object_node.children[0].color + " "])
                        + np.random.choice([object_node.children[0].name, object_node.children[0].category, *object_node.children[0].aliases])
                        + " and the "
                        + np.random.choice(["", object_node.children[1].color + " "])
                        + np.random.choice([object_node.children[1].name, object_node.children[1].category, *object_node.children[1].aliases])
                    )
                    if object_node.children[0].obj_id != object_node.children[1].obj_id
                    else (
                        "in the middle of "
                        + np.random.choice(["the ", "two "])
                        + np.random.choice(["", object_node.children[0].color + " "])
                        + np.random.choice([object_node.children[0].name, object_node.children[0].category, *object_node.children[0].aliases])
                        + "s"
                    )
                )
            elif object_node.locationComponent == self.components.location.relative_to_multiple.row_order:
                orderWords = ["first", "second", "third", "fourth"]
                leftOrRightWords = ["left", "right"]
                leftOrRightWord = np.random.choice(leftOrRightWords)
                if leftOrRightWord == "left":
                    expression += np.random.choice(clause_words) + orderWords[object_node.order - 1] + " from left"
                else:
                    expression += orderWords[4 - object_node.order] + " from right"
        elif object_node.locationComponent in self.components.location.absolute.all_components:
            absoluteLocationWords = get_absolute_location_words(object_node.locationComponent)
            expression += np.random.choice(clause_words) + np.random.choice(absoluteLocationWords)
        elif object_node.locationComponent == self.components.location.unspecified:
            # check attribute component
            if object_node.attributeComponent == self.components.attribute.indirect.same_color:
                phrases = ["that has the same color as", "which has the same color as", "whose color is the same as"]
                expression += np.random.choice(phrases) + " " + self.get_expression(object_node.parent, is_parent=True, get_parent_color=False)[0]
            elif object_node.attributeComponent == self.components.attribute.indirect.not_color:
                phrases = ["that is not", "which is not", "whose color is not"]
                expression += np.random.choice(phrases) + " " + object_node.children[0].color
            elif object_node.attributeComponent == self.components.attribute.unspecified:
                pass
            elif object_node.attributeComponent == self.components.attribute.direct.color:
                pass
            else:
                raise RuntimeError(f"attributeComponent is not a valid attribute component:\n{object_node}\nscne graph:\n{self}")
        else:
            raise RuntimeError(f"locationComponent is not a valid location component:\n{object_node}\nscne graph:\n{self}")
        return expression

    def get_simple_referring_expressions(self) -> list[str]:
        """
        Get a list of simple referring expressions for each object node with expression components in the scene graph.
        Object nodes come from `self.referringExpressionStructures.keys()`.

        Returns:
            list[str]: A list of simple referring expressions for each object node with expression components in the scene graph.
        """
        # TODO: color attribute only for now
        expressions = []
        for objectNode in self.expression_structures.keys():
            expressions.append(self.get_expression(objectNode))
        return expressions

    def get_complex_referring_expressions(self) -> list[str]:
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

        def make_sentence_from_list(expressionList: list[str]) -> str:
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
        for simpleExpression in self.get_simple_referring_expressions():
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
            complexExpressions.append(make_sentence_from_list(expressionList))
        self.generated_expressions = complexExpressions
        return complexExpressions

    def check_forbidden_expressions(
        self, obj: list, location_component: int, attribute_component: int, refer_component: int, coordinate: tuple
    ) -> bool:
        expressionComponentsForCheck = [
            {self.components.attribute.direct.color: obj[3], self.components.refer.name: obj[1]},
            {self.components.attribute.direct.color: obj[3], self.components.refer.category: obj[2]},
            {self.components.attribute.direct.color: obj[3], self.components.refer.ambiguous: True},
            {self.components.attribute.unspecified: True, self.components.refer.name: obj[1]},
            {self.components.attribute.unspecified: True, self.components.refer.category: obj[2]},
            {self.components.attribute.unspecified: True, self.components.refer.ambiguous: True},
        ]
        if (
            location_component == self.components.location.unspecified
            or location_component == self.components.location.relative_to_multiple.row_order
            or attribute_component in self.components.attribute.indirect.all_components
        ):
            for item in expressionComponentsForCheck:
                for forbiddenExpressionComponentsRegion in self.banned_components_global:
                    if Misc.DictIn(item, forbiddenExpressionComponentsRegion):
                        return False

        for item in expressionComponentsForCheck:
            for forbiddenExpressionComponentsRegion in self.banned_components_regional[coordinate]:
                if Misc.DictIn(item, forbiddenExpressionComponentsRegion):
                    return False
        return True

    def __str__(self) -> str:
        string = "Map:\n"
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                string += f"{(self.map[i, j].name if self.map[i, j] is not None else '.'):^30}"
            string += "\n"
        string += "\nObjectNodes:\n"
        expressions = self.get_complex_referring_expressions() if self.generated_expressions is None else self.generated_expressions
        for objectNode in self.object_nodes:
            string += f"\n{objectNode}"
            for i, node_with_expressions in enumerate(self.expression_structures.keys()):
                if node_with_expressions == objectNode:
                    string += f"Expression: {expressions[i]}\n"
        return string

    def save(self, file_path: str):
        """
        Save the scene graph to a file.\n

        Args:
            file_path (str): The file name (with path) of the scene graph.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    def count_components(self) -> dict:
        """
        Count the number of each component in the scene graph.

        Returns:
            dict: A dictionary with keys as component names and values as the number of each component.
        """
        component_counts = {}
        for object_node in self.expression_structures.keys():
            for component in object_node.referringExpressionStructure:
                if component not in component_counts:
                    component_counts[self.components.get_component_name(component)] = 1
                else:
                    component_counts[self.components.get_component_name(component)] += 1
        return component_counts

    @staticmethod
    def write_scene_graph_to_file(scene_graph: SceneGraph, save_path: str, file_name_prefix: str) -> int:
        """
        Save the scene graph to a folder that may contain multiple scene graphs.\n
        The file is named like "`file_name_prefix`_00000012.pkl".

        Args:
            scene_graph (SceneGraph): The scene graph.
            save_path (str): The folder path.
            name_suffix (str): The file name suffix.

        Returns:
            int: The index of created scene graph file.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        num_existing_files = len(os.listdir(save_path))
        file_name = f"{file_name_prefix}_{num_existing_files:08d}.pkl"
        scene_graph.save(os.path.join(save_path, file_name))
        return num_existing_files

    @staticmethod
    def load(file_path: str) -> SceneGraph:
        """
        Load a scene graph from a file.\n

        Args:
            file_path (str): The file path of the scene graph.

        Returns:
            SceneGraph: The scene graph.
        """
        with open(file_path, "rb") as f:
            sceneGraph = pickle.load(f)
        return sceneGraph
