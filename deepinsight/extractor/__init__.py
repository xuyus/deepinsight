import os
from ._shared import FuncCallConvention, register_func_call_convention
from ._blocks import BlockCallConvention, patching_block_with_type_hook
from ._blocks_hierarchy import (
    BlockHierarchyCallConvention,
    patching_block_and_operator_with_hierarchy_hook,
)
from ._operators import OperatorCallConvention, patching_operator_hook

register_func_call_convention(
    OperatorCallConvention.CONVENTION_TYPE, OperatorCallConvention
)
register_func_call_convention(BlockCallConvention.CONVENTION_TYPE, BlockCallConvention)
register_func_call_convention(
    BlockHierarchyCallConvention.CONVENTION_TYPE, BlockHierarchyCallConvention
)
