class Misc:
    @staticmethod
    def DictIn(smaller_dict: dict, larger_dict: dict) -> bool:
        """
        Check if the larger dictionary contains the smaller dictionary.

        Args:
            smaller_dict (dict): The smaller dictionary.
            larger_dict (dict): The larger dictionary.

        Returns:
            bool: True if the larger dictionary contains the smaller dictionary, False otherwise.
        """
        for key, value in smaller_dict.items():
            if key not in larger_dict or larger_dict[key] != value:
                return False
        return True
