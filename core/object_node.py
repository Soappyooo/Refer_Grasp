from __future__ import annotations
import core.scene_graph
from utils.misc import Misc


class ObjectNode:
    """
    [WIP] An object node in the scene graph.\n

    """

    def __init__(
        self,
        scene: core.scene_graph.SceneGraph,
        obj: str,
        name: str,
        category: str,
        coordinate: tuple[int, int],
        color: str,
        size: float,
        referringExpressionStructure: list,
        parent: ObjectNode = None,
        order: int = -1,
        aliases: list = None,
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
        self.scene_id = len(self.scene.object_nodes)
        self.obj_id = obj
        self.name = name
        self.aliases = aliases
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

    def Register(self):
        self.scene.object_nodes.append(self)
        self.scene.map[self.coordinate] = self
        if self.referringExpressionStructure != []:
            self.scene.expression_structures[self] = [self.locationComponent, self.attributeComponent, self.objectReferComponent]
        self.UpdateForbiddenExpressionComponents()

    def UpdateForbiddenExpressionComponents(self):
        """
        [WIP] Update the forbidden expression components map of the scene graph.\n
        For each coordinate in the forbidden expression components map, there is a list of dictionaries of forbidden expression components.
        The dictionaries contain the expression component and value pair, e.g. `{color: "red", objectRefer: "can"}`.
        Before a new object node is created at a coordinate, the forbidden expression components at this coordinate should be checked.
        """
        if self.referringExpressionStructure == []:
            return
        # TODO: attribute dict if more attribute components are added
        objectReferDict = {
            self.scene.components.refer.name: self.name,
            self.scene.components.refer.category: self.category,
            self.scene.components.refer.ambiguous: True,
        }

        targetCoordinates = self.scene.get_coordinates(self.locationComponent, self.parent, self.order)
        # TODO revise if more attribute
        if (
            self.attributeComponent == self.scene.components.attribute.direct.color
            or self.attributeComponent == self.scene.components.attribute.indirect.same_color
        ):
            components = {
                self.scene.components.attribute.direct.color: self.color,
                self.objectReferComponent: objectReferDict[self.objectReferComponent],
            }
        elif (
            self.attributeComponent == self.scene.components.attribute.unspecified
            or self.attributeComponent == self.scene.components.attribute.indirect.not_color
        ):
            components = {
                self.scene.components.attribute.unspecified: True,
                self.objectReferComponent: objectReferDict[self.objectReferComponent],
            }
        if components not in self.scene.banned_components_global:
            self.scene.banned_components_global.append(components)

        if (
            self.locationComponent in self.scene.components.location.relative_to_single.all_components
            or self.locationComponent in self.scene.components.location.absolute.all_components
        ):
            for coordinate in targetCoordinates:
                if Misc.DictIn(components, self.scene.banned_components_regional[coordinate]) is False:
                    self.scene.banned_components_regional[coordinate].append(components)

        if (
            self.locationComponent == self.scene.components.location.relative_to_multiple.row_order
            or self.locationComponent == self.scene.components.location.unspecified
            or self.attributeComponent in self.scene.components.attribute.indirect.all_components
        ):
            targetCoordinates = self.scene.get_coordinates(self.scene.components.location.unspecified, self.parent, self.order, filter=True)
            for coordinate in targetCoordinates:
                if Misc.DictIn(components, self.scene.banned_components_regional[coordinate]) is False:
                    self.scene.banned_components_regional[coordinate].append(components)

    def AddChild(
        self,
        obj: str,
        name: str,
        category: str,
        coordinate: tuple[int, int],
        color: str,
        size: float,
        referringExpressionStructure: list,
        aliases: list = None,
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
        child = ObjectNode(self.scene, obj, name, category, coordinate, color, size, referringExpressionStructure, parent=self, aliases=aliases)
        self.children.append(child)
        return child

    def SatisfyExpression(self, expression: dict) -> bool:
        objectReferDict = {
            self.scene.components.refer.name: self.name,
            self.scene.components.refer.category: self.category,
            self.scene.components.refer.ambiguous: True,
        }
        if self.scene.components.attribute.direct.color in expression and expression[self.scene.components.attribute.direct.color] != self.color:
            return False
        if (
            self.scene.components.attribute.indirect.not_color in expression
            and expression[self.scene.components.attribute.indirect.not_color] == self.color
        ):
            return False

        for objectRefer in objectReferDict.keys():
            if objectRefer in expression and expression[objectRefer] != objectReferDict[objectRefer]:
                return False
        return True

    def __str__(self) -> str:
        return (
            f"ObjectNode:{self.name}(scene_id={self.scene_id}, obj_id={self.obj_id}, coordinate={self.coordinate}, color={self.color}, size={self.size}, aliases={self.aliases}\n"
            + f"           parent={f'{self.parent.name} at {self.parent.coordinate}' if self.parent is not None else None},\n"
            + f"           children={[f'{child.name} at {child.coordinate}' for child in self.children]},\n"
            + f"           locationComponent={self.scene.components.get_component_name(self.locationComponent)},\n"
            + f"           attributeComponent={self.scene.components.get_component_name(self.attributeComponent)},\n"
            + f"           objectReferComponent={self.scene.components.get_component_name(self.objectReferComponent)})\n"
        )

    def __repr__(self) -> str:
        return self.__str__()
