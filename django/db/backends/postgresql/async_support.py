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

    def __getattr__(self, attr):
        return getattr(self.db, attr)

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

    def get_compiler(self, query, *, elide_empty=True):
        return self.ops.compiler(query.compiler)(
            query,
            self,
            self.alias,
            elide_empty,
        )

    async def execute_compiler(self, compiler, *, result_type):
        from django.core.exceptions import EmptyResultSet
        from django.db.models.sql.constants import (
            CURSOR,
            MULTI,
            NO_RESULTS,
            ROW_COUNT,
            SINGLE,
        )

        result_type = result_type or NO_RESULTS
        try:
            sql, params = compiler.as_sql()
            if not sql:
                raise EmptyResultSet
        except EmptyResultSet:
            if result_type == MULTI:
                return []
            return None

        cursor = self.cursor()
        try:
            await cursor.execute(sql, params)
        except Exception as e:
            try:
                await cursor.close()
            except self.db.Database.Error:
                raise e from None
            raise

        if result_type == ROW_COUNT:
            try:
                return cursor.rowcount
            finally:
                await cursor.close()
        if result_type == CURSOR:
            return cursor
        if result_type == NO_RESULTS:
            await cursor.close()
            return None

        try:
            if result_type == SINGLE:
                row = await cursor.fetchone()
                if row:
                    return row[0 : compiler.col_count]
                return row

            rows = await cursor.fetchall()
            if compiler.has_extra_select and compiler.col_count is not None:
                rows = [row[0 : compiler.col_count] for row in rows]
            return [rows]
        finally:
            await cursor.close()

    async def execute_insert_query(self, query, returning_fields=None):
        from django.db.models import AutoField

        compiler = self.get_compiler(query)
        assert not (
            returning_fields
            and len(query.objs) != 1
            and not self.features.can_return_rows_from_bulk_insert
        )
        opts = query.get_meta()
        compiler.returning_fields = returning_fields
        cols = []
        async with self.cursor() as cursor:
            for sql, params in compiler.as_sql():
                await cursor.execute(sql, params)
            if not compiler.returning_fields:
                return []
            obj_len = len(query.objs)
            if (
                self.features.can_return_rows_from_bulk_insert
                and obj_len > 1
            ) or (
                self.features.can_return_columns_from_insert and obj_len == 1
            ):
                rows = await cursor.fetchall()
                cols = [field.get_col(opts.db_table) for field in compiler.returning_fields]
            elif returning_fields and isinstance(
                returning_field := returning_fields[0], AutoField
            ):
                cols = [returning_field.get_col(opts.db_table)]
                rows = [
                    (
                        self.ops.last_insert_id(
                            cursor,
                            opts.db_table,
                            returning_field.column,
                        ),
                    )
                ]
            else:
                return []

        converters = compiler.get_converters(cols)
        if converters:
            rows = compiler.apply_converters(rows, converters)
        return list(rows)

    async def execute_update_query(self, query, returning_fields=None):
        from django.db.models.sql.constants import ROW_COUNT

        compiler = self.get_compiler(query)
        if returning_fields is None:
            row_count = await self.execute_compiler(compiler, result_type=ROW_COUNT)
            is_empty = row_count is None
            row_count = row_count or 0

            for related_query in query.get_related_updates():
                aux_row_count = await self.execute_update_query(related_query)
                if is_empty and aux_row_count:
                    row_count = aux_row_count
                    is_empty = False
            return row_count

        if query.get_related_updates():
            raise NotImplementedError(
                "Update returning is not implemented for queries with related updates."
            )
        if not returning_fields or not self.features.can_return_rows_from_update:
            row_count = await self.execute_update_query(query)
            return [()] * row_count

        compiler.returning_fields = returning_fields
        async with self.cursor() as cursor:
            sql, params = compiler.as_sql()
            await cursor.execute(sql, params)
            rows = await cursor.fetchall()

        opts = query.get_meta()
        cols = [field.get_col(opts.db_table) for field in compiler.returning_fields]
        converters = compiler.get_converters(cols)
        if converters:
            rows = compiler.apply_converters(rows, converters)
        return list(rows)

    async def count_queryset(self, queryset):
        from django.core.exceptions import EmptyResultSet

        compiler = self.get_compiler(queryset.query)
        try:
            sql, params = compiler.as_sql()
            if not sql:
                raise EmptyResultSet
        except EmptyResultSet:
            return 0

        async with self.cursor() as cursor:
            await cursor.execute(f"SELECT COUNT(*) FROM ({sql}) subquery", params)
            row = await cursor.fetchone()
        return row[0]

    async def has_results_queryset(self, queryset):
        from django.db.models.sql.constants import SINGLE

        compiler = self.get_compiler(queryset.query.exists())
        return bool(await self.execute_compiler(compiler, result_type=SINGLE))

    async def fetch_model_queryset(self, queryset):
        import operator
        from itertools import chain
        from weakref import ref as weak_ref

        from django.db.models.query import ModelIterable, get_related_populators
        from django.db.models.sql.constants import MULTI

        if queryset._iterable_class is not ModelIterable:
            raise NotImplementedError(
                "Native async PostgreSQL get() currently supports only model querysets."
            )

        db = queryset.db
        compiler = self.get_compiler(queryset.query)
        results = await self.execute_compiler(compiler, result_type=MULTI)
        select, klass_info, annotation_col_map = (
            compiler.select,
            compiler.klass_info,
            compiler.annotation_col_map,
        )
        model_cls = klass_info["model"]
        select_fields = klass_info["select_fields"]
        model_fields_start, model_fields_end = select_fields[0], select_fields[-1] + 1
        init_list = [
            f[0].target.attname for f in select[model_fields_start:model_fields_end]
        ]
        related_populators = get_related_populators(
            klass_info,
            select,
            db,
            queryset._fetch_mode,
        )
        known_related_objects = [
            (
                field,
                related_objs,
                attnames := [
                    (
                        field.attname
                        if from_field == "self"
                        else queryset.model._meta.get_field(from_field).attname
                    )
                    for from_field in field.from_fields
                ],
                operator.attrgetter(*attnames),
            )
            for field, related_objs in queryset._known_related_objects.items()
        ]
        fields = [s[0] for s in select[0 : compiler.col_count]]
        rows = chain.from_iterable(results)
        converters = compiler.get_converters(fields)
        if converters:
            rows = compiler.apply_converters(rows, converters)
        if compiler.has_composite_fields(fields):
            rows = compiler.composite_fields_to_tuples(rows, fields)

        objs = []
        peers = []
        for row in rows:
            obj = model_cls.from_db(
                db,
                init_list,
                row[model_fields_start:model_fields_end],
                fetch_mode=queryset._fetch_mode,
            )
            if queryset._fetch_mode.track_peers:
                peers.append(weak_ref(obj))
                obj._state.peers = peers
            for rel_populator in related_populators:
                rel_populator.populate(row, obj)
            if annotation_col_map:
                for attr_name, col_pos in annotation_col_map.items():
                    setattr(obj, attr_name, row[col_pos])
            for field, rel_objs, rel_attnames, rel_getter in known_related_objects:
                if field.is_cached(obj):
                    continue
                if any(attname not in obj.__dict__ for attname in rel_attnames):
                    continue
                rel_obj_id = rel_getter(obj)
                try:
                    rel_obj = rel_objs[rel_obj_id]
                except KeyError:
                    pass
                else:
                    setattr(obj, field.name, rel_obj)
            objs.append(obj)
        return objs

    def _make_savepoint_id(self):
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        task_ident = str(id(current_task)) if current_task is not None else "sync"
        self.savepoint_state += 1
        return f"s{task_ident}_x{self.savepoint_state}"
