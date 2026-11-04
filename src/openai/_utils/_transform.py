from __future__ import annotations

import io
import base64
import pathlib
from typing import Any, Mapping, TypeVar, cast
from datetime import date, datetime
from typing_extensions import Literal, get_args, override, get_type_hints as _get_type_hints

import anyio
import pydantic

from ._utils import (
    is_list,
    is_given,
    lru_cache,
    is_mapping,
    is_iterable,
    is_sequence,
)
from .._files import is_base64_file_input
from ._compat import get_origin, is_typeddict
from ._typing import (
    is_list_type,
    is_union_type,
    is_iterable_type,
    is_required_type,
    is_sequence_type,
    is_annotated_type,
    is_not_required_type,
    strip_annotated_type,
)

_T = TypeVar("_T")


# TODO: support for drilling globals() and locals()
# TODO: ensure works correctly with forward references in all cases


PropertyFormat = Literal["iso8601", "base64", "custom"]


class PropertyInfo:
    """Metadata class to be used in Annotated types to provide information about a given type.

    For example:

    class MyParams(TypedDict):
        account_holder_name: Annotated[str, PropertyInfo(alias='accountHolderName')]

    This means that {'account_holder_name': 'Robert'} will be transformed to {'accountHolderName': 'Robert'} before being sent to the API.
    """

    alias: str | None
    format: PropertyFormat | None
    format_template: str | None
    discriminator: str | None

    def __init__(
        self,
        *,
        alias: str | None = None,
        format: PropertyFormat | None = None,
        format_template: str | None = None,
        discriminator: str | None = None,
    ) -> None:
        self.alias = alias
        self.format = format
        self.format_template = format_template
        self.discriminator = discriminator

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alias='{self.alias}', format={self.format}, format_template='{self.format_template}', discriminator='{self.discriminator}')"


def maybe_transform(
    data: object,
    expected_type: object,
) -> Any | None:
    """Wrapper over `transform()` that allows `None` to be passed.

    See `transform()` for more details.
    """
    if data is None:
        return None
    return transform(data, expected_type)


# Wrapper over _transform_recursive providing fake types
def transform(
    data: _T,
    expected_type: object,
) -> _T:
    """Transform dictionaries based off of type information from the given type, for example:

    ```py
    class Params(TypedDict, total=False):
        card_id: Required[Annotated[str, PropertyInfo(alias="cardID")]]


    transformed = transform({"card_id": "<my card ID>"}, Params)
    # {'cardID': '<my card ID>'}
    ```

    Any keys / data that does not have type information given will be included as is.

    It should be noted that the transformations that this function does are not represented in the type system.
    """
    transformed = _transform_recursive(data, annotation=cast(type, expected_type))
    return cast(_T, transformed)


@lru_cache(maxsize=8096)
def _get_annotated_type(type_: type) -> type | None:
    """If the given type is an `Annotated` type then it is returned, if not `None` is returned.

    This also unwraps the type when applicable, e.g. `Required[Annotated[T, ...]]`
    """
    if is_required_type(type_) or is_not_required_type(type_):
        # Unwrap Required or NotRequired to expose the Annotated metadata
        type_ = get_args(type_)[0]

    if is_annotated_type(type_):
        return type_

    return None


def _maybe_transform_key(key: str, type_: type) -> str:
    """Transform the given `data` based on the annotations provided in `type_`.

    Note: this function only looks at `Annotated` types that contain `PropertyInfo` metadata.
    """
    annotated_type = _get_annotated_type(type_)
    if annotated_type is None:
        # no `Annotated` definition for this type, no transformation needed
        return key

    # ignore the first argument as it is the actual type
    annotations = get_args(annotated_type)[1:]
    for annotation in annotations:
        if isinstance(annotation, PropertyInfo) and annotation.alias is not None:
            return annotation.alias

    return key


def _no_transform_needed(annotation: type) -> bool:
    return annotation == float or annotation == int


@lru_cache(maxsize=8096)
def _contains_property_info(type_: object) -> bool | None:
    """Whether the given type's annotations contain any `PropertyInfo` metadata.

    Returns:
        * `True` when the type carries `PropertyInfo` metadata (aliases,
          formats or discriminators) anywhere in its structure;
        * `False` when it provably does not **and** transforming matching data
          is an identity function for every possible value (i.e. the type is
          "closed": no bare containers, no `Any`/`object`, nothing that can
          hide a pydantic model needing `model_dump`);
        * `None` when the answer cannot be decided from the type alone
          (bare `list`/`dict`/`Any`/`object`), in which case callers must
          fall back to the regular recursive transform.

    The result is cached per type, so that data matching a metadata-free type
    can skip the full walk of the data entirely (see GH openai/openai-python
    #4021 for why that matters for large bodies).
    """
    if isinstance(type_, PropertyInfo):
        return True

    if type_ is Any or type_ is object:
        return None

    if isinstance(type_, type) and issubclass(type_, pydantic.BaseModel):
        # models are model_dump'ed (possibly excluding `__api_exclude__`
        # fields), which is not an identity function
        return None

    if is_annotated_type(type_):
        args = get_args(type_)
        if any(isinstance(arg, PropertyInfo) for arg in args[1:]):
            return True
        return _contains_property_info(args[0])

    if is_required_type(type_) or is_not_required_type(type_):
        return _contains_property_info(get_args(type_)[0])

    if is_typeddict(type_):
        undecided = False
        for field_type in get_type_hints(type_, include_extras=True).values():
            result = _contains_property_info(field_type)
            if result is True:
                return True
            if result is None:
                undecided = True
        if undecided:
            return None
        return False

    origin = get_origin(type_)
    if origin is None:
        if type_ in (list, dict, tuple, set, frozenset):
            # bare container classes: their contents are undecidable (they
            # may hold pydantic models that need model_dump)
            return None
        return False

    if is_union_type(type_):
        results = [_contains_property_info(subtype) for subtype in get_args(type_)]
        if any(r is True for r in results):
            return True
        if any(r is None for r in results):
            return None
        return False

    # container types: for list[T] / Iterable[T] / Sequence[T] the item type
    # (args[0]) can carry metadata; for dict[K, V] only the value type (and
    # anything after the key arg) can. A bare container (no args) leaves the
    # contents undecidable.
    args = get_args(type_)
    if not args:
        return None
    if origin == dict:
        args = args[1:]
    results = [_contains_property_info(arg) for arg in args]
    if any(r is True for r in results):
        return True
    if any(r is None for r in results):
        return None
    return False


def _transform_recursive(
    data: object,
    *,
    annotation: type,
    inner_type: type | None = None,
) -> object:
    """Transform the given data against the expected type.

    Args:
        annotation: The direct type annotation given to the particular piece of data.
            This may or may not be wrapped in metadata types, e.g. `Required[T]`, `Annotated[T, ...]` etc

        inner_type: If applicable, this is the "inside" type. This is useful in certain cases where the outside type
            is a container type such as `List[T]`. In that case `inner_type` should be set to `T` so that each entry in
            the list can be transformed using the metadata from the container type.

            Defaults to the same value as the `annotation` argument.
    """
    from .._compat import model_dump

    if inner_type is None:
        inner_type = annotation

    stripped_type = strip_annotated_type(inner_type)
    origin = get_origin(stripped_type) or stripped_type
    if is_typeddict(stripped_type) and is_mapping(data):
        return _transform_typeddict(data, stripped_type)

    if origin == dict and is_mapping(data):
        args = get_args(stripped_type)
        items_type = args[1] if len(args) > 1 else object
        if not (
            _contains_property_info(items_type) is False
            # the outer `Annotated[..., PropertyInfo(format=...)]` applies to
            # every value in the mapping, so it must be checked too
            and _contains_property_info(annotation) is False
        ):
            # metadata may be reachable from here — walk the mapping
            return {key: _transform_recursive(value, annotation=items_type) for key, value in data.items()}
        # no `PropertyInfo` metadata reachable from here — keep the original
        # mapping without walking it (GH#4021)
        return data

    if (
        # List[T]
        (is_list_type(stripped_type) and is_list(data))
        # Iterable[T]
        or (is_iterable_type(stripped_type) and is_iterable(data) and not isinstance(data, str))
        # Sequence[T]
        or (is_sequence_type(stripped_type) and is_sequence(data) and not isinstance(data, str))
    ):
        # dicts are technically iterable, but it is an iterable on the keys of the dict and is not usually
        # intended as an iterable, so we don't transform it.
        if isinstance(data, dict):
            return cast(object, data)

        args = get_args(stripped_type)
        inner_type = cast(type, args[0]) if args else object
        if _no_transform_needed(inner_type):
            # for some types there is no need to transform anything, so we can get a small
            # perf boost from skipping that work.

            # but we still need to convert to a list to ensure the data is json-serializable
            if is_list(data):
                return data
            return list(data)

        if is_list(data) and (
            _contains_property_info(inner_type) is False
            # the outer `Annotated[..., PropertyInfo(format=...)]` applies to
            # every entry of the list, so it must be checked too
            and _contains_property_info(annotation) is False
        ):
            # no `PropertyInfo` metadata reachable from here — keep the
            # original list without walking it (GH#4021)
            return data

        return [_transform_recursive(d, annotation=annotation, inner_type=inner_type) for d in data]

    if is_union_type(stripped_type):
        # For union types we run the transformation against all subtypes to ensure that everything is transformed.

        # TODO: there may be edge cases where the same normalized field name will transform to two different names
        # in different subtypes.
        for subtype in get_args(stripped_type):
            data = _transform_recursive(data, annotation=annotation, inner_type=subtype)
        return data

    if isinstance(data, pydantic.BaseModel):
        return model_dump(data, exclude_unset=True, mode="json", exclude=getattr(data, "__api_exclude__", None))

    annotated_type = _get_annotated_type(annotation)
    if annotated_type is None:
        return data

    # ignore the first argument as it is the actual type
    annotations = get_args(annotated_type)[1:]
    for annotation in annotations:
        if isinstance(annotation, PropertyInfo) and annotation.format is not None:
            return _format_data(data, annotation.format, annotation.format_template)

    return data


def _format_data(data: object, format_: PropertyFormat, format_template: str | None) -> object:
    if isinstance(data, (date, datetime)):
        if format_ == "iso8601":
            return data.isoformat()

        if format_ == "custom" and format_template is not None:
            return data.strftime(format_template)

    if format_ == "base64" and is_base64_file_input(data):
        binary: str | bytes | None = None

        if isinstance(data, pathlib.Path):
            binary = data.read_bytes()
        elif isinstance(data, io.IOBase):
            binary = data.read()

            if isinstance(binary, str):
                binary = binary.encode()

        if not isinstance(binary, bytes):
            raise RuntimeError(f"Could not read bytes from {data}; Received {type(binary)}")

        return base64.b64encode(binary).decode("ascii")

    return data


def _transform_typeddict(
    data: Mapping[str, object],
    expected_type: type,
) -> Mapping[str, object]:
    if _contains_property_info(expected_type) is False:
        # the type is provably metadata-free and closed, so the transform is
        # an identity function for any data matching it — return the input
        # as-is instead of walking every value (GH#4021)
        return data

    result: dict[str, object] = {}
    annotations = get_type_hints(expected_type, include_extras=True)
    for key, value in data.items():
        if not is_given(value):
            # we don't need to include omitted values here as they'll
            # be stripped out before the request is sent anyway
            continue

        type_ = annotations.get(key)
        if type_ is None:
            # we do not have a type annotation for this field, leave it as is
            result[key] = value
        else:
            result[_maybe_transform_key(key, type_)] = _transform_recursive(value, annotation=type_)
    return result


async def async_maybe_transform(
    data: object,
    expected_type: object,
) -> Any | None:
    """Wrapper over `async_transform()` that allows `None` to be passed.

    See `async_transform()` for more details.
    """
    if data is None:
        return None
    return await async_transform(data, expected_type)


async def async_transform(
    data: _T,
    expected_type: object,
) -> _T:
    """Transform dictionaries based off of type information from the given type, for example:

    ```py
    class Params(TypedDict, total=False):
        card_id: Required[Annotated[str, PropertyInfo(alias="cardID")]]


    transformed = transform({"card_id": "<my card ID>"}, Params)
    # {'cardID': '<my card ID>'}
    ```

    Any keys / data that does not have type information given will be included as is.

    It should be noted that the transformations that this function does are not represented in the type system.
    """
    transformed = await _async_transform_recursive(data, annotation=cast(type, expected_type))
    return cast(_T, transformed)


async def _async_transform_recursive(
    data: object,
    *,
    annotation: type,
    inner_type: type | None = None,
) -> object:
    """Transform the given data against the expected type.

    Args:
        annotation: The direct type annotation given to the particular piece of data.
            This may or may not be wrapped in metadata types, e.g. `Required[T]`, `Annotated[T, ...]` etc

        inner_type: If applicable, this is the "inside" type. This is useful in certain cases where the outside type
            is a container type such as `List[T]`. In that case `inner_type` should be set to `T` so that each entry in
            the list can be transformed using the metadata from the container type.

            Defaults to the same value as the `annotation` argument.
    """
    from .._compat import model_dump

    if inner_type is None:
        inner_type = annotation

    stripped_type = strip_annotated_type(inner_type)
    origin = get_origin(stripped_type) or stripped_type
    if is_typeddict(stripped_type) and is_mapping(data):
        return await _async_transform_typeddict(data, stripped_type)

    if origin == dict and is_mapping(data):
        args = get_args(stripped_type)
        items_type = args[1] if len(args) > 1 else object
        if not (
            _contains_property_info(items_type) is False
            # the outer `Annotated[..., PropertyInfo(format=...)]` applies to
            # every value in the mapping, so it must be checked too
            and _contains_property_info(annotation) is False
        ):
            # metadata may be reachable from here — walk the mapping
            # note: mirrors the upstream implementation, which reuses the sync
            # recursive transform here
            return {key: _transform_recursive(value, annotation=items_type) for key, value in data.items()}
        # no `PropertyInfo` metadata reachable from here — keep the original
        # mapping without walking it (GH#4021)
        return data

    if (
        # List[T]
        (is_list_type(stripped_type) and is_list(data))
        # Iterable[T]
        or (is_iterable_type(stripped_type) and is_iterable(data) and not isinstance(data, str))
        # Sequence[T]
        or (is_sequence_type(stripped_type) and is_sequence(data) and not isinstance(data, str))
    ):
        # dicts are technically iterable, but it is an iterable on the keys of the dict and is not usually
        # intended as an iterable, so we don't transform it.
        if isinstance(data, dict):
            return cast(object, data)

        args = get_args(stripped_type)
        inner_type = cast(type, args[0]) if args else object
        if _no_transform_needed(inner_type):
            # for some types there is no need to transform anything, so we can get a small
            # perf boost from skipping that work.

            # but we still need to convert to a list to ensure the data is json-serializable
            if is_list(data):
                return data
            return list(data)

        if is_list(data) and (
            _contains_property_info(inner_type) is False
            # the outer `Annotated[..., PropertyInfo(format=...)]` applies to
            # every entry of the list, so it must be checked too
            and _contains_property_info(annotation) is False
        ):
            # no `PropertyInfo` metadata reachable from here — keep the
            # original list without walking it (GH#4021)
            return data

        return [await _async_transform_recursive(d, annotation=annotation, inner_type=inner_type) for d in data]

    if is_union_type(stripped_type):
        # For union types we run the transformation against all subtypes to ensure that everything is transformed.

        # TODO: there may be edge cases where the same normalized field name will transform to two different names
        # in different subtypes.
        for subtype in get_args(stripped_type):
            data = await _async_transform_recursive(data, annotation=annotation, inner_type=subtype)
        return data

    if isinstance(data, pydantic.BaseModel):
        return model_dump(data, exclude_unset=True, mode="json", exclude=getattr(data, "__api_exclude__", None))

    annotated_type = _get_annotated_type(annotation)
    if annotated_type is None:
        return data

    # ignore the first argument as it is the actual type
    annotations = get_args(annotated_type)[1:]
    for annotation in annotations:
        if isinstance(annotation, PropertyInfo) and annotation.format is not None:
            return await _async_format_data(data, annotation.format, annotation.format_template)

    return data


async def _async_format_data(data: object, format_: PropertyFormat, format_template: str | None) -> object:
    if isinstance(data, (date, datetime)):
        if format_ == "iso8601":
            return data.isoformat()

        if format_ == "custom" and format_template is not None:
            return data.strftime(format_template)

    if format_ == "base64" and is_base64_file_input(data):
        binary: str | bytes | None = None

        if isinstance(data, pathlib.Path):
            binary = await anyio.Path(data).read_bytes()
        elif isinstance(data, io.IOBase):
            binary = data.read()

            if isinstance(binary, str):
                binary = binary.encode()

        if not isinstance(binary, bytes):
            raise RuntimeError(f"Could not read bytes from {data}; Received {type(binary)}")

        return base64.b64encode(binary).decode("ascii")

    return data


async def _async_transform_typeddict(
    data: Mapping[str, object],
    expected_type: type,
) -> Mapping[str, object]:
    if _contains_property_info(expected_type) is False:
        # the type is provably metadata-free and closed, so the transform is
        # an identity function for any data matching it — return the input
        # as-is instead of walking every value (GH#4021)
        return data

    result: dict[str, object] = {}
    annotations = get_type_hints(expected_type, include_extras=True)
    for key, value in data.items():
        if not is_given(value):
            # we don't need to include omitted values here as they'll
            # be stripped out before the request is sent anyway
            continue

        type_ = annotations.get(key)
        if type_ is None:
            # we do not have a type annotation for this field, leave it as is
            result[key] = value
        else:
            result[_maybe_transform_key(key, type_)] = await _async_transform_recursive(value, annotation=type_)
    return result


@lru_cache(maxsize=8096)
def get_type_hints(
    obj: Any,
    globalns: dict[str, Any] | None = None,
    localns: Mapping[str, Any] | None = None,
    include_extras: bool = False,
) -> dict[str, Any]:
    return _get_type_hints(obj, globalns=globalns, localns=localns, include_extras=include_extras)
