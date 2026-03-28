import asyncio

from django.db.backends.utils import debug_transaction


class AsyncCursorWrapper:
    def __init__(self, cursor, db):
        self.cursor = cursor
        self.db = db

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
        with self.db.wrap_database_errors:
            if params is None:
                await self.cursor.execute(sql, **kwargs)
            else:
                await self.cursor.execute(sql, params, **kwargs)
        return self

    async def executemany(self, sql, param_list, *, returning=False):
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
        self.savepoint_state = 0

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

    @property
    def autocommit(self):
        return self.connection.autocommit

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
        )

    async def execute(self, query, params=None, *, prepare=None, binary=False):
        kwargs = {"binary": binary}
        if prepare is not None:
            kwargs["prepare"] = prepare
        with self.db.wrap_database_errors:
            cursor = await self.connection.execute(query, params, **kwargs)
        return AsyncCursorWrapper(cursor, self.db)

    async def commit(self):
        with debug_transaction(self.db, "COMMIT"), self.db.wrap_database_errors:
            await self.connection.commit()

    async def rollback(self):
        with debug_transaction(self.db, "ROLLBACK"), self.db.wrap_database_errors:
            await self.connection.rollback()

    async def close(self):
        if self.closed:
            return
        try:
            with self.db.wrap_database_errors:
                await self.connection.close()
        finally:
            self.connection = None

    async def savepoint(self):
        if not self.db.features.uses_savepoints or self.autocommit:
            return
        sid = self._make_savepoint_id()
        async with self.cursor() as cursor:
            await cursor.execute(self.db.ops.savepoint_create_sql(sid))
        return sid

    async def savepoint_rollback(self, sid):
        if not self.db.features.uses_savepoints or self.autocommit:
            return
        async with self.cursor() as cursor:
            await cursor.execute(self.db.ops.savepoint_rollback_sql(sid))

    async def savepoint_commit(self, sid):
        if not self.db.features.uses_savepoints or self.autocommit:
            return
        async with self.cursor() as cursor:
            await cursor.execute(self.db.ops.savepoint_commit_sql(sid))

    def _make_savepoint_id(self):
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        task_ident = str(id(current_task)) if current_task is not None else "sync"
        self.savepoint_state += 1
        return f"s{task_ident}_x{self.savepoint_state}"
