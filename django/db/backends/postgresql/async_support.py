import asyncio

from django.db import DatabaseError, Error
from django.db.backends.utils import debug_transaction
from django.db.transaction import TransactionManagementError


class AsyncAtomic:
    def __init__(self, connection, savepoint, durable):
        self.connection = connection
        self.savepoint = savepoint
        self.durable = durable
        self._from_testcase = False

    async def __aenter__(self):
        connection = self.connection

        if (
            self.durable
            and connection.atomic_blocks
            and not connection.atomic_blocks[-1]._from_testcase
        ):
            raise RuntimeError(
                "A durable atomic block cannot be nested within another "
                "atomic block."
            )

        if not connection.in_atomic_block:
            connection.commit_on_exit = True
            connection.needs_rollback = False
            connection.rollback_exc = None
            if not connection.get_autocommit():
                connection.in_atomic_block = True
                connection.commit_on_exit = False

        if connection.in_atomic_block:
            if self.savepoint and not connection.needs_rollback:
                sid = await connection.savepoint()
                connection.savepoint_ids.append(sid)
            else:
                connection.savepoint_ids.append(None)
        else:
            await connection._set_autocommit(False)
            connection.in_atomic_block = True

        if connection.in_atomic_block:
            connection.atomic_blocks.append(self)

        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        connection = self.connection

        if connection.in_atomic_block:
            connection.atomic_blocks.pop()

        if connection.savepoint_ids:
            sid = connection.savepoint_ids.pop()
        else:
            connection.in_atomic_block = False
            sid = None

        try:
            if connection.closed_in_transaction:
                pass
            elif exc_type is None and not connection.needs_rollback:
                if connection.in_atomic_block:
                    if sid is not None:
                        try:
                            await connection.savepoint_commit(sid)
                        except DatabaseError:
                            try:
                                await connection.savepoint_rollback(sid)
                                await connection.savepoint_commit(sid)
                            except Error:
                                connection.needs_rollback = True
                            raise
                else:
                    try:
                        await connection._commit()
                    except DatabaseError:
                        try:
                            await connection._rollback()
                        except Error:
                            await connection.close()
                        raise
            else:
                connection.needs_rollback = False
                if connection.in_atomic_block:
                    if sid is None:
                        connection.needs_rollback = True
                        if exc_value is not None:
                            connection.rollback_exc = exc_value
                    else:
                        try:
                            await connection.savepoint_rollback(sid)
                            await connection.savepoint_commit(sid)
                        except Error:
                            connection.needs_rollback = True
                            if exc_value is not None:
                                connection.rollback_exc = exc_value
                else:
                    try:
                        await connection._rollback()
                    except Error:
                        await connection.close()
        finally:
            if not connection.in_atomic_block:
                if connection.closed_in_transaction:
                    connection.connection = None
                else:
                    await connection._set_autocommit(True)
            elif not connection.savepoint_ids and not connection.commit_on_exit:
                if connection.closed_in_transaction:
                    connection.connection = None
                else:
                    connection.in_atomic_block = False


class AsyncCursorWrapper:
    def __init__(self, cursor, db, transaction_connection=None):
        self.cursor = cursor
        self.db = db
        self.transaction_connection = transaction_connection

    def __getattr__(self, attr):
        return getattr(self.cursor, attr)

    def __aiter__(self):
        return self.cursor.__aiter__()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        try:
            await self.close()
        except self.db.Database.Error:
            pass

    async def execute(self, sql, params=None, *, prepare=None, binary=None):
        kwargs = {}
        if prepare is not None:
            kwargs["prepare"] = prepare
        if binary is not None:
            kwargs["binary"] = binary
        if self.transaction_connection is not None:
            self.transaction_connection.validate_no_broken_transaction()
        with self.db.wrap_database_errors:
            if params is None:
                await self.cursor.execute(sql, **kwargs)
            else:
                await self.cursor.execute(sql, params, **kwargs)
        return self

    async def executemany(self, sql, param_list, *, returning=False):
        if self.transaction_connection is not None:
            self.transaction_connection.validate_no_broken_transaction()
        with self.db.wrap_database_errors:
            await self.cursor.executemany(sql, param_list, returning=returning)

    async def fetchone(self):
        with self.db.wrap_database_errors:
            return await self.cursor.fetchone()

    async def fetchmany(self, size=0):
        with self.db.wrap_database_errors:
            if size:
                return await self.cursor.fetchmany(size)
            return await self.cursor.fetchmany()

    async def fetchall(self):
        with self.db.wrap_database_errors:
            return await self.cursor.fetchall()

    async def close(self):
        with self.db.wrap_database_errors:
            await self.cursor.close()


class AsyncPostgreSQLConnection:
    """
    A PostgreSQL-only async connection helper used for native async substrate
    experiments. This intentionally lives beside Django's sync connection path
    instead of trying to share its state.
    """

    def __init__(self, db, connection):
        self.db = db
        self.connection = connection
        self.autocommit = connection.autocommit
        self.in_atomic_block = False
        self.savepoint_state = 0
        self.savepoint_ids = []
        self.atomic_blocks = []
        self.commit_on_exit = True
        self.needs_rollback = False
        self.rollback_exc = None
        self.closed_in_transaction = False

    @classmethod
    async def connect(cls, db, *, autocommit=None):
        conn_params = db.get_async_connection_params()
        connection = await db.get_new_async_connection(
            conn_params,
            autocommit=autocommit,
        )
        return cls(db, connection)

    @property
    def closed(self):
        return self.connection is None or self.connection.closed

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self.closed:
            return
        try:
            if exc_type is None and not self.autocommit:
                await self.commit()
            elif exc_type is not None and not self.autocommit:
                await self.rollback()
        finally:
            await self.close()

    def cursor(self, name=None, *, binary=False, scrollable=None, withhold=False):
        return AsyncCursorWrapper(
            self.connection.cursor(
                name=name or "",
                binary=binary,
                scrollable=scrollable,
                withhold=withhold,
            ),
            self.db,
            transaction_connection=self,
        )

    async def execute(self, query, params=None, *, prepare=None, binary=False):
        kwargs = {"binary": binary}
        if prepare is not None:
            kwargs["prepare"] = prepare
        self.validate_no_broken_transaction()
        with self.db.wrap_database_errors:
            cursor = await self.connection.execute(query, params, **kwargs)
        return AsyncCursorWrapper(cursor, self.db, transaction_connection=self)

    def get_autocommit(self):
        return self.autocommit

    async def set_autocommit(self, autocommit):
        self.validate_no_atomic_block()
        await self._set_autocommit(autocommit)

    async def commit(self):
        self.validate_no_atomic_block()
        await self._commit()
        self.needs_rollback = False
        self.rollback_exc = None

    async def _commit(self):
        with debug_transaction(self.db, "COMMIT"), self.db.wrap_database_errors:
            await self.connection.commit()

    async def rollback(self):
        self.validate_no_atomic_block()
        await self._rollback()
        self.needs_rollback = False
        self.rollback_exc = None

    async def _rollback(self):
        with debug_transaction(self.db, "ROLLBACK"), self.db.wrap_database_errors:
            await self.connection.rollback()

    async def close(self):
        if self.closed:
            return
        try:
            with self.db.wrap_database_errors:
                await self.connection.close()
        finally:
            if self.in_atomic_block:
                self.closed_in_transaction = True
                self.needs_rollback = True
            else:
                self.connection = None

    async def _set_autocommit(self, autocommit):
        if self.closed:
            return
        if not autocommit:
            with debug_transaction(self.db, "BEGIN"), self.db.wrap_database_errors:
                await self.connection.set_autocommit(False)
        else:
            with self.db.wrap_database_errors:
                await self.connection.set_autocommit(True)
        self.autocommit = autocommit

    async def savepoint(self):
        if not self.db.features.uses_savepoints or self.get_autocommit():
            return
        sid = self._make_savepoint_id()
        async with self.cursor() as cursor:
            await cursor.execute(self.db.ops.savepoint_create_sql(sid))
        return sid

    async def savepoint_rollback(self, sid):
        if not self.db.features.uses_savepoints or self.get_autocommit():
            return
        async with self.cursor() as cursor:
            await cursor.execute(self.db.ops.savepoint_rollback_sql(sid))

    async def savepoint_commit(self, sid):
        if not self.db.features.uses_savepoints or self.get_autocommit():
            return
        async with self.cursor() as cursor:
            await cursor.execute(self.db.ops.savepoint_commit_sql(sid))

    def atomic(self, *, savepoint=True, durable=False):
        return AsyncAtomic(self, savepoint, durable)

    def get_rollback(self):
        if not self.in_atomic_block:
            raise TransactionManagementError(
                "The rollback flag doesn't work outside of an 'atomic' block."
            )
        return self.needs_rollback

    def set_rollback(self, rollback):
        if not self.in_atomic_block:
            raise TransactionManagementError(
                "The rollback flag doesn't work outside of an 'atomic' block."
            )
        self.needs_rollback = rollback
        if not rollback:
            self.rollback_exc = None

    def validate_no_atomic_block(self):
        if self.in_atomic_block:
            raise TransactionManagementError(
                "This is forbidden when an 'atomic' block is active."
            )

    def validate_no_broken_transaction(self):
        if self.needs_rollback:
            raise TransactionManagementError(
                "An error occurred in the current transaction. You can't "
                "execute queries until the end of the 'atomic' block."
            ) from self.rollback_exc

    def _make_savepoint_id(self):
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        task_ident = str(id(current_task)) if current_task is not None else "sync"
        self.savepoint_state += 1
        return f"s{task_ident}_x{self.savepoint_state}"
