"""
Custom transforms adapter to make a list of transforms callable.
This bridges the gap between our code and the existing dataset structure.
"""

class TransformsAdapter:
    """
    Adapter class that makes a list of transforms behave like a single callable transform.
    """
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list if transforms_list else []
        
    def __call__(self, imgs):
        """
        Apply all transforms in sequence to the input images.
        """
        result = imgs
        for transform in self.transforms_list:
            result = transform(result)
        return result