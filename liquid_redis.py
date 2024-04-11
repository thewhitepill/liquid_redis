from __future__ import annotations

import asyncio
import inspect

from asyncio import Lock, Task

from inspect import signature
from typing import (
    Generic,
    Hashable,
    Optional,
    Type,
    TypeAlias,
    TypeVar,
    Union
)

from liquid import Reducer, Store
from pydantic import BaseModel, TypeAdapter, field_validator, field_serializer
from redis.asyncio import Redis, WatchError
from redis.asyncio.client import PubSub


__all__ = (
    "ConcurrencyError",
    "InvalidStateError",
    "RedisStore",

    "redis_store_factory"
)


A = TypeVar("A")
S = TypeVar("S", bound=Hashable)


class InvalidStateError(Exception):
    pass


class ConcurrencyError(Exception):
    pass


def _get_model_generic_args(model_type: type[BaseModel]) -> tuple[type, ...]:
    return model_type.__pydantic_generic_metadata__["args"]


def _get_reducer_types(reducer: Reducer[S, A]) -> tuple[Type[S], Type[A]]:
    types = [p.annotation for p in signature(reducer).parameters.values()]

    if len(types) != 2:
        raise TypeError("Reducer must have exactly two parameters")

    if inspect._empty in types:
        raise TypeError("Reducer parameters must have type annotations")

    return tuple(types)


class _ActionContainer(BaseModel, Generic[A]):
    action: A
    previous_state_hash: int
    updated_state_hash: int

    @staticmethod
    def channel_name(namespace: str) -> str:
        return f"{namespace}:actions"


class _StateContainer(BaseModel, Generic[S]):
    state: S
    state_hash: int

    @field_serializer("state")
    @classmethod
    def serialize_state(cls, value: S) -> str:
        state_type: type[S] = _get_model_generic_args(cls)[0]

        return TypeAdapter(state_type).serialize_json(value)

    @field_serializer("state_hash")
    @classmethod
    def serialize_state_hash(cls, value: int) -> str:
        return str(value)

    @field_validator("state", mode="before")
    @classmethod
    def validate_state(cls, value: Union[S, bytes]) -> S:
        state_type: type[S] = _get_model_generic_args(cls)[0]

        if isinstance(value, state_type):
            return value

        assert isinstance(value, bytes)

        return TypeAdapter(state_type).validate_json(value)

    @field_validator("state_hash", mode="before")
    @classmethod
    def validate_state_hash(cls, value: Union[int, bytes]) -> int:
        if isinstance(value, int):
            return value

        return int(value.decode())

    @staticmethod
    def state_key(namespace: str) -> str:
        return f"{namespace}:state"

    @staticmethod
    def state_hash_key(namespace: str) -> str:
        return f"{namespace}:state_hash"


async def _get_state_container(
    cls: TypeAlias,
    client: Redis,
    namespace: str
) -> Optional[_StateContainer[S]]:
    if not await client.exists(_StateContainer.state_hash_key(namespace)):
        return None

    state, state_hash = await client.mget(
        _StateContainer.state_key(namespace),
        _StateContainer.state_hash_key(namespace)
    )

    return _StateContainer[cls](state=state, state_hash=state_hash)


async def _set_state_container(
    container: _StateContainer[S],
    current_local_state_hash: Optional[int],
    client: Redis,
    namespace: str
) -> None:
    async with client.pipeline(transaction=True) as pipe:
        await pipe.watch(_StateContainer.state_hash_key(namespace))

        if current_local_state_hash:
            current_remote_state_hash = _StateContainer.validate_state_hash( # type: ignore[call-arg]
                await pipe.get(
                    _StateContainer.state_hash_key(namespace)
                )
            )

            if current_remote_state_hash != current_local_state_hash:
                raise ConcurrencyError

        pipe.multi()

        data = container.model_dump()

        pipe.mset({
            _StateContainer.state_key(namespace): data["state"],
            _StateContainer.state_hash_key(namespace): data["state_hash"]
        })

        try:
            await pipe.execute()
        except WatchError:
            raise ConcurrencyError


class RedisStore(Store[S, A, S]):
    _state_type: Type[S]
    _action_type: Type[A]

    _reducer: Reducer[S, A]
    _initial_state: S

    _state: Optional[S]
    _state_hash: Optional[int]

    _lock: Lock

    _redis_client: Optional[Redis]
    _redis_namespace: Optional[str]
    _redis_pubsub_handler_task: Optional[Task]

    def __init__(
        self,
        state_type: Type[S],
        action_type: Type[A],
        reducer: Reducer[S, A],
        initial_state: S
    ) -> None:
        self._state_type = state_type
        self._action_type = action_type

        self._reducer = reducer
        self._initial_state = initial_state

        self._state = None
        self._state_hash = None

        self._lock = Lock()

        self._redis_client = None
        self._redis_namespace = None
        self._redis_pubsub_handler_task = None

    async def _pubsub_handler(self, pubsub: PubSub) -> None:
        assert self._state is not None
        assert self._state_hash is not None
        assert self._redis_client is not None
        assert self._redis_namespace is not None

        action_type = self._action_type

        while True:
            message = await pubsub.get_message()

            if message is None:
                continue

            data = message["data"]
            action_container: _ActionContainer[A] = \
                _ActionContainer[action_type].model_validate_json(data) # type: ignore[valid-type]

            async with self._lock:
                is_fresh = self._state_hash == \
                    action_container.previous_state_hash

                if not is_fresh:
                    state_container: Optional[_StateContainer[S]] = \
                        await _get_state_container(
                            self._state_type,
                            self._redis_client,
                            self._redis_namespace
                        )

                    assert state_container is not None

                    self._state = state_container.state
                    self._state_hash = state_container.state_hash

                    return

                action = action_container.action
                self._state = self._reducer(self._state, action)
                self._state_hash = action_container.updated_state_hash

    async def bind(self, redis_client: Redis, redis_namespace: str) -> None:
        if self._state is not None:
            raise InvalidStateError

        state_type = self._state_type

        async with self._lock:
            self._redis_client = redis_client
            self._redis_namespace = redis_namespace

            state_container: Optional[_StateContainer[S]] = \
                await _get_state_container(
                    state_type,
                    self._redis_client,
                    self._redis_namespace
                )

            if state_container is None:
                self._state = self._initial_state
                self._state_hash = hash(self._state)

                state_container = _StateContainer[state_type]( # type: ignore[valid-type]
                    state=self._state,
                    state_hash=self._state_hash
                )

                await _set_state_container(
                    state_container,
                    None,
                    self._redis_client,
                    self._redis_namespace
                )
            else:
                self._state = state_container.state
                self._state_hash = state_container.state_hash

            async with self._redis_client.pubsub() as pubsub:
                await pubsub.subscribe(
                    _ActionContainer.channel_name(self._redis_namespace),
                    ignore_subscribe_messages=True
                )

                self._redis_pubsub_handler_task = asyncio.create_task(
                    self._pubsub_handler(pubsub)
                )

    async def unbind(self) -> None:
        if self._state is None:
            raise InvalidStateError

        assert self._redis_pubsub_handler_task is not None

        async with self._lock:
            self._redis_pubsub_handler_task.cancel()

            self._state = None
            self._state_hash = None

            self._redis_client = None
            self._redis_namespace = None
            self._redis_pubsub_handler_task = None

    async def dispatch(self, action: A) -> S: # type: ignore[override]
        if not self._state:
            raise InvalidStateError

        assert self._state_hash is not None
        assert self._redis_client is not None
        assert self._redis_namespace is not None

        state_type = self._state_type
        action_type = self._action_type
        state_container: Optional[_StateContainer[S]]

        async with self._lock:
            previous_state_hash = self._state_hash

            self._state = self._reducer(self._state, action)
            self._state_hash = hash(self._state)
            state_container = _StateContainer[state_type]( # type: ignore[valid-type]
                state=self._state,
                state_hash=self._state_hash
            )

            try:
                await _set_state_container(
                    state_container,
                    previous_state_hash,
                    self._redis_client,
                    self._redis_namespace
                )
            except ConcurrencyError:
                state_container = await _get_state_container(
                    state_type,
                    self._redis_client,
                    self._redis_namespace
                )

                assert state_container is not None

                self._state = state_container.state
                self._state_hash = state_container.state_hash

                raise

            action_container = _ActionContainer[action_type]( # type: ignore[valid-type]
                action=action,
                previous_state_hash=previous_state_hash,
                updated_state_hash=self._state_hash
            )

            await self._redis_client.publish(
                _ActionContainer.channel_name(self._redis_namespace),
                action_container.model_dump_json()
            )

            assert self._state is not None

            return self._state

    async def get_state(self) -> S:
        if self._state is None:
            raise InvalidStateError

        async with self._lock:
            return self._state


def redis_store_factory(
    reducer: Reducer[S, A],
    initial_state: S
) -> RedisStore[S, A]:
    state_type, action_type = _get_reducer_types(reducer)

    return RedisStore(
        state_type,
        action_type,
        reducer,
        initial_state
    )
