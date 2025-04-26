# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
import inspect
from typing import Any, Callable, Optional, Union, get_args
from datetime import date

from google.genai import types
from typing_extensions import override

from ._automatic_function_calling_util import build_function_declaration
from .base_tool import BaseTool
from .tool_context import ToolContext


class FunctionTool(BaseTool):
  """A tool that wraps a user-defined Python function.

  Attributes:
    func: The function to wrap.
  """

  def __init__(self, func: Callable[..., Any]):
    super().__init__(name=func.__name__, description=func.__doc__)
    self.func = func

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    function_decl = types.FunctionDeclaration.model_validate(
        build_function_declaration(
            func=self.func,
            # The model doesn't understand the function context.
            # input_stream is for streaming tool
            ignore_params=['tool_context', 'input_stream'],
            variant=self._api_variant,
        )
    )

    return function_decl

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    args_to_call = args.copy()
    signature = inspect.signature(self.func)

    # I don't really know where to put this
    # If we're also gonna support datetime and other non-primitive types in the future maybe just break this into a bigger helper method
    # if we have any enum values, we need to convert them to the enum member
    for param_name, param in signature.parameters.items():
      if param_name in args_to_call:
        annotation = param.annotation
        # Handle Union[EnumType, None] (Optional) case
        if hasattr(annotation, '__origin__') and annotation.__origin__ is Union:
          types = get_args(annotation)
          # Get the type in the Unionthat's not None
          actual_type = next((t for t in types if t is not type(None)), None)
          if actual_type:
            annotation = actual_type
        
        if inspect.isclass(annotation) and issubclass(annotation, Enum):
          enum_class = annotation
          string_value = args_to_call[param_name]
          args_to_call[param_name] = enum_class[string_value]
        
        # Handle dates (from date library)
        elif annotation is date and isinstance(args_to_call[param_name], str):
          try:
            args_to_call[param_name] = date.fromisoformat(args_to_call[param_name])
          except ValueError as e:
            return {'error': f"Invalid date format for parameter '{param_name}'. Expected YYYY-MM-DD format: {str(e)}"}

    if 'tool_context' in signature.parameters:
      args_to_call['tool_context'] = tool_context

    # Before invoking the function, we check for if the list of args passed in
    # has all the mandatory arguments or not.
    # If the check fails, then we don't invoke the tool and let the Agent know
    # that there was a missing a input parameter. This will basically help
    # the underlying model fix the issue and retry.
    mandatory_args = self._get_mandatory_args()
    missing_mandatory_args = [
        arg for arg in mandatory_args if arg not in args_to_call
    ]

    if missing_mandatory_args:
      missing_mandatory_args_str = '\n'.join(missing_mandatory_args)
      error_str = f"""Invoking `{self.name}()` failed as the following mandatory input parameters are not present:
{missing_mandatory_args_str}
You could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters."""
      return {'error': error_str}

    if inspect.iscoroutinefunction(self.func):
      return await self.func(**args_to_call) or {}
    else:
      return self.func(**args_to_call) or {}

  # TODO(hangfei): fix call live for function stream.
  async def _call_live(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
      invocation_context,
  ) -> Any:
    args_to_call = args.copy()
    signature = inspect.signature(self.func)
    if (
        self.name in invocation_context.active_streaming_tools
        and invocation_context.active_streaming_tools[self.name].stream
    ):
      args_to_call['input_stream'] = invocation_context.active_streaming_tools[
          self.name
      ].stream
    if 'tool_context' in signature.parameters:
      args_to_call['tool_context'] = tool_context
    async for item in self.func(**args_to_call):
      yield item

  def _get_mandatory_args(
      self,
  ) -> list[str]:
    """Identifies mandatory parameters (those without default values) for a function.

    Returns:
      A list of strings, where each string is the name of a mandatory parameter.
    """
    signature = inspect.signature(self.func)
    mandatory_params = []

    for name, param in signature.parameters.items():
      # A parameter is mandatory if:
      # 1. It has no default value (param.default is inspect.Parameter.empty)
      # 2. It's not a variable positional (*args) or variable keyword (**kwargs) parameter
      #
      # For more refer to: https://docs.python.org/3/library/inspect.html#inspect.Parameter.kind
      if param.default == inspect.Parameter.empty and param.kind not in (
          inspect.Parameter.VAR_POSITIONAL,
          inspect.Parameter.VAR_KEYWORD,
      ):
        mandatory_params.append(name)

    return mandatory_params
