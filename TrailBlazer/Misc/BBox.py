"""
"""

import pdb
import torch

def compute_bbox_LRTB_HW(dim_x, dim_y, box_ratios, use_int=False, margin=0.01):

    # assert (box_ratios[1] < box_ratios[3]), "the boundary top ratio should be less than bottom"
    # assert (box_ratios[0] < box_ratios[2]), "the boundary left ratio should be less than right"
    # left = ((box_ratios[0] + margin) * dim_x) #.int()
    # right = ((box_ratios[2] - margin) * dim_x) #.int()
    # top = ((box_ratios[1] + margin) * dim_y) #.int()
    # bottom = ((box_ratios[3] - margin) * dim_y) #.int()
    
    # height = bottom - top
    # width = right - left
    # if height == 0:
    #     height = 1
    # if width == 0:
    #     width = 1

    # return left, right, top, bottom, height, width

    # """
    assert (box_ratios[1] < box_ratios[3]), f"the boundary `top` ratio {box_ratios[1]} should be less than `bottom` {box_ratios[3]}"
    assert (box_ratios[0] < box_ratios[2]), f"the boundary `left` ratio {box_ratios[0]} should be less than `right` {box_ratios[2]}"

    # adding a 1% margin also addresses negative box percent values
    left_ = ((box_ratios[0] + margin) * dim_x) 
    right_ = ((box_ratios[2] - margin) * dim_x) 
    top_ = ((box_ratios[1] + margin) * dim_y) 
    bottom_ = ((box_ratios[3] - margin) * dim_y) 
    # pdb.set_trace()

    # REMOVE debugging
    # if ((box_ratios[0] + margin).detach() < 0) or ((box_ratios[2] - margin).detach() > 1) or ((box_ratios[1] + margin) < 0).detach() or ((box_ratios[3] - margin).detach() > 1):
    #     pdb.set_trace()
     
    if use_int:
        left = ((box_ratios[0] + margin) * dim_x).int()
        right = ((box_ratios[2] - margin) * dim_x).int()
        top = ((box_ratios[1] + margin) * dim_y).int()
        bottom = ((box_ratios[3] - margin) * dim_y).int()

        # using `round` as pytorch still provides backprop through it [as a straight-through estimator]
        # left = torch.round((box_ratios[0] + margin) * dim_x)
        # right = torch.round((box_ratios[2] - margin) * dim_x)
        # top = torch.round((box_ratios[1] + margin) * dim_y)
        # bottom = torch.round((box_ratios[3] - margin) * dim_y)
        # pdb.set_trace()
    else:
        # continous box values
        left, right, top, bottom = left_, right_, top_, bottom_
        
    # ----------
    height = bottom - top
    width = right - left

    if height == 0:
        height = 1
        print(f'------- height computes 0!')
        # pdb.set_trace()
    if width == 0:
        print(f'------- width computes 0!')
        width = 1
        # pdb.set_trace()

    return left, right, top, bottom, height, width
    # """
    
class BoundingBox:
    """A rectangular bounding box determines the directed regions."""

    def __init__(self, dim_x, dim_y, box_ratios, margin=0.01):
        """
        Args:
            resolution(int): the resolution of the 2d spatial input
            box_ratios(List[float]):
        Returns:
        """
        assert (
            box_ratios[1] < box_ratios[3]
        ), "the boundary top ratio should be less than bottom"
        assert (
            box_ratios[0] < box_ratios[2]
        ), "the boundary left ratio should be less than right"
        self.left = int((box_ratios[0] + margin) * dim_x)
        self.right = int((box_ratios[2] - margin) * dim_x)
        self.top = int((box_ratios[1] + margin) * dim_y)
        self.bottom = int((box_ratios[3] - margin) * dim_y)
        # pdb.set_trace()
        

        self.height = self.bottom - self.top
        self.width = self.right - self.left

        # NOTE: Original Traiblazer fix for 0-collapse in small layer resolutions
        if self.height == 0:
            self.height = 1
            self.bottom = self.top + 1 # address the fix; add 1 to avoid slicing error later
        if self.width == 0:
            self.width = 1
            self.right = self.left + 1 # address the fix; add 1 to avoid slicing error later
        
        # pdb.set_trace()
        # print(f'left {self.left}, right {self.right}, top {self.top}, bottom {self.bottom}, height {self.height}, width {self.width}')
        # self.left, self.right, self.top, self.bottom, self.height, self.width 

    def sliced_tensor_in_bbox(self, tensor: torch.tensor) -> torch.tensor:
        """ slicing the tensor with bbox area

        Args:
            tensor(torch.tensor): the original tensor in 4d
        Returns:
            (torch.tensor): the reduced tensor inside bbox
        """
        return tensor[:, self.top : self.bottom, self.left : self.right, :]

    def mask_reweight_out_bbox(
        self, tensor: torch.tensor, value: float = 0.0
    ) -> torch.tensor:
        """reweighting value outside bbox

        Args:
            tensor(torch.tensor): the original tensor in 4d
            value(float): reweighting factor default with 0.0
        Returns:
            (torch.tensor): the reweighted tensor
        """
        mask = torch.ones_like(tensor).to(tensor.device) * value
        mask[:, self.top : self.bottom, self.left : self.right, :] = 1
        return tensor * mask

    def mask_reweight_in_bbox(
        self, tensor: torch.tensor, value: float = 0.0
    ) -> torch.tensor:
        """reweighting value within bbox

        Args:
            tensor(torch.tensor): the original tensor in 4d
            value(float): reweighting factor default with 0.0
        Returns:
            (torch.tensor): the reweighted tensor
        """
        mask = torch.ones_like(tensor).to(tensor.device)
        mask[:, self.top : self.bottom, self.left : self.right, :] = value
        return tensor * mask

    def __str__(self):
        """it prints Box(L:%d, R:%d, T:%d, B:%d) for better ingestion"""
        return f"Box(L:{self.left}, R:{self.right}, T:{self.top}, B:{self.bottom})"

    def __rerp__(self):
        """ """
        return f"Box(L:{self.left}, R:{self.right}, T:{self.top}, B:{self.bottom})"


if __name__ == "__main__":
    # Example: second quadrant
    input_res = 32
    left = 0.0
    top = 0.0
    right = 0.5
    bottom = 0.5
    box_ratios = [left, top, right, bottom]
    bbox = BoundingBox(resolution=input_res, box_ratios=box_ratios)

    print(bbox)
    # Box(L:0, R:16, T:0, B:16)
