import functools
import numpy as np
from typing import Union


class ExpressionComponentsBase:
    @functools.cached_property
    def all_components(self) -> list[int]:
        """
        List of all components in the specified expression components class.

        Returns:
            list[int]: A list of all components in the specified expression components class.
        """
        result = []
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, int):
                result.append(attr_value)
            # elif hasattr(attr_value, "__dict__"):
            #     result.extend(attr_value.allComponents)
            elif isinstance(attr_value, ExpressionComponentsBase):
                result.extend(attr_value.all_components)
        return result

    def get_random_component(self) -> int:
        """
        Get a random component in the specified expression components class.

        Returns:
            int: A random component in the specified expression components class.
        """
        return np.random.choice(self.all_components)

    @functools.cache
    def get_component_name(self, component_id: int) -> str:
        """
        Get the name of the specified component for a given component id.

        Args:
            component_id (int): The component id.

        Returns:
            str: The component name, None if the component id is not found.
        """
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, int):
                if attr_value == component_id:
                    return attr_name
            elif isinstance(attr_value, ExpressionComponentsBase):
                result = attr_value.get_component_name(component_id)
                if result is not None:
                    return f"{attr_name}.{result}"
        return None


class ExpressionComponents(ExpressionComponentsBase):
    """
    Assign an integer to each expression component, can be perceived as a struct.
    e.g. `components.location.relative_to_single.front` is 0. `components.refer.name` is 700.

    """

    def __init__(self) -> None:
        self.location = self.Location()
        self.attribute = self.Attribute()
        self.refer = self.ObjectRefer()

    @staticmethod
    def randomly_select_from(*args: Union[int, list[int]]) -> int:
        """
        Randomly select a component from the given components. e.g. `randomly_select_from(0, 1, [2, 3])` will randomly choose from `[0, 1, 2, 3]`.

        Returns:
            int: A randomly selected component.
        """
        choices = []
        for arg in args:
            if isinstance(arg, list):
                choices += arg
            else:
                choices.append(arg)
        return np.random.choice(choices)

    class Location(ExpressionComponentsBase):
        """0-399"""

        def __init__(self) -> None:
            self.relative_to_single = self.RelativeLocationToSingleObject()
            self.relative_to_multiple = self.RelativeLocationToMultipleObjects()
            self.absolute = self.AbsoluteLocation()
            self.unspecified = 300

        class RelativeLocationToSingleObject(ExpressionComponentsBase):
            def __init__(self) -> None:
                self.front = 0
                self.rear = 1
                self.left = 2
                self.right = 3
                self.front_left = 4
                self.front_right = 5
                self.rear_left = 6
                self.rear_right = 7
                self.near = 8
                # TODO: top bottom not

        class RelativeLocationToMultipleObjects(ExpressionComponentsBase):
            def __init__(self) -> None:
                self.middle = 100
                self.row_order = 101
                # TODO: columnOrder front rear left right...

        class AbsoluteLocation(ExpressionComponentsBase):
            def __init__(self) -> None:
                self.frontmost = 200
                self.rearmost = 201
                self.leftmost = 202
                self.rightmost = 203
                self.front_leftmost = 204
                self.front_rightmost = 205
                self.rear_leftmost = 206
                self.rear_rightmost = 207

    class Attribute(ExpressionComponentsBase):
        """400-699"""

        def __init__(self) -> None:
            self.direct = self.DirectAttribute()
            self.indirect = self.IndirectAttribute()
            self.unspecified = 600

        class DirectAttribute(ExpressionComponentsBase):
            def __init__(self) -> None:
                self.color = 400
                # self.size = 401
                # TODO shape material

        class IndirectAttribute(ExpressionComponentsBase):
            def __init__(self) -> None:
                self.not_color = 500
                self.same_color = 501
                # TODO shape material
                pass

    class ObjectRefer(ExpressionComponentsBase):
        def __init__(self) -> None:
            self.name = 700
            self.category = 701
            self.ambiguous = 702
