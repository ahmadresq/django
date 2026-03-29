import asyncio
import unittest

from django.db import DatabaseError, connection, transaction
from django.db.models import Count, Max, Prefetch
from django.test import TransactionTestCase, modify_settings

from .models import Character, CharFieldModel, Line, Scene

try:
    from django.db.backends.postgresql.psycopg_any import is_psycopg3
except ImportError:
    is_psycopg3 = False


class AsyncAtomicTestError(Exception):
    pass


@unittest.skipUnless(connection.vendor == "postgresql", "PostgreSQL specific tests")
@unittest.skipUnless(is_psycopg3, "Native async PostgreSQL support requires psycopg 3")
@modify_settings(INSTALLED_APPS={"append": "django.contrib.postgres"})
class PostgreSQLAsyncSupportTests(TransactionTestCase):
    available_apps = ["django.contrib.postgres", "postgres_tests"]

    async def _reset_table(self, async_connection, table_name):
        await async_connection.execute(f"DROP TABLE IF EXISTS {table_name}")
        await async_connection.execute(
            f"CREATE TABLE {table_name} (id serial PRIMARY KEY, value integer)"
        )

    async def _count_rows(self, async_connection, table_name):
        async with async_connection.cursor() as cursor:
            await cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row = await cursor.fetchone()
        return row

    async def test_new_async_connection_execute_and_fetchone(self):
        async with await connection.new_async_connection() as async_connection:
            async with async_connection.cursor() as cursor:
                await cursor.execute("SELECT 1")
                row = await cursor.fetchone()

        self.assertEqual(row, (1,))

    async def test_new_async_connection_commit_and_rollback(self):
        async with await connection.new_async_connection(
            autocommit=False
        ) as async_connection:
            async with async_connection.cursor() as cursor:
                await cursor.execute(
                    "CREATE TEMP TABLE async_support_commit_rollback (value integer)"
                )
                await cursor.execute(
                    "INSERT INTO async_support_commit_rollback (value) VALUES (1)"
                )
            await async_connection.commit()

            async with async_connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT COUNT(*) FROM async_support_commit_rollback"
                )
                committed_count = await cursor.fetchone()
                await cursor.execute(
                    "INSERT INTO async_support_commit_rollback (value) VALUES (2)"
                )
            await async_connection.rollback()

            async with async_connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT COUNT(*) FROM async_support_commit_rollback"
                )
                rolled_back_count = await cursor.fetchone()

        self.assertEqual(committed_count, (1,))
        self.assertEqual(rolled_back_count, (1,))

    async def test_new_async_connection_savepoint_rollback(self):
        async with await connection.new_async_connection(
            autocommit=False
        ) as async_connection:
            async with async_connection.cursor() as cursor:
                await cursor.execute(
                    "CREATE TEMP TABLE async_support_savepoint (value integer)"
                )

            sid = await async_connection.savepoint()

            async with async_connection.cursor() as cursor:
                await cursor.execute(
                    "INSERT INTO async_support_savepoint (value) VALUES (1)"
                )

            await async_connection.savepoint_rollback(sid)

            async with async_connection.cursor() as cursor:
                await cursor.execute("SELECT COUNT(*) FROM async_support_savepoint")
                row = await cursor.fetchone()

        self.assertEqual(row, (0,))

    async def test_async_atomic_commits_transaction(self):
        table_name = "async_support_atomic_commit"
        async with await connection.new_async_connection() as async_connection:
            await self._reset_table(async_connection, table_name)

            async with async_connection.atomic():
                await async_connection.execute(
                    f"INSERT INTO {table_name} (value) VALUES (1)"
                )

            row = await self._count_rows(async_connection, table_name)
            await async_connection.execute(f"DROP TABLE {table_name}")

        self.assertEqual(row, (1,))

    async def test_async_atomic_rolls_back_on_exception(self):
        table_name = "async_support_atomic_rollback"
        async with await connection.new_async_connection() as async_connection:
            await self._reset_table(async_connection, table_name)

            with self.assertRaisesMessage(AsyncAtomicTestError, "Oops"):
                async with async_connection.atomic():
                    await async_connection.execute(
                        f"INSERT INTO {table_name} (value) VALUES (1)"
                    )
                    raise AsyncAtomicTestError("Oops")

            row = await self._count_rows(async_connection, table_name)
            await async_connection.execute(f"DROP TABLE {table_name}")

        self.assertEqual(row, (0,))

    async def test_async_atomic_nested_savepoint_rolls_back_inner_block(self):
        table_name = "async_support_atomic_savepoint"
        async with await connection.new_async_connection() as async_connection:
            await self._reset_table(async_connection, table_name)

            async with async_connection.atomic():
                await async_connection.execute(
                    f"INSERT INTO {table_name} (value) VALUES (1)"
                )
                with self.assertRaisesMessage(AsyncAtomicTestError, "Oops"):
                    async with async_connection.atomic():
                        await async_connection.execute(
                            f"INSERT INTO {table_name} (value) VALUES (2)"
                        )
                        raise AsyncAtomicTestError("Oops")
                inner_row = await self._count_rows(async_connection, table_name)

            outer_row = await self._count_rows(async_connection, table_name)
            await async_connection.execute(f"DROP TABLE {table_name}")

        self.assertEqual(inner_row, (1,))
        self.assertEqual(outer_row, (1,))

    async def test_async_atomic_savepoint_false_marks_transaction_for_rollback(self):
        table_name = "async_support_atomic_broken"
        msg = (
            "An error occurred in the current transaction. You can't execute "
            "queries until the end of the 'atomic' block."
        )
        async with await connection.new_async_connection() as async_connection:
            await self._reset_table(async_connection, table_name)

            async with async_connection.atomic():
                await async_connection.execute(
                    f"INSERT INTO {table_name} (value) VALUES (1)"
                )
                with self.assertRaisesMessage(AsyncAtomicTestError, "Oops"):
                    async with async_connection.atomic(savepoint=False):
                        await async_connection.execute(
                            f"INSERT INTO {table_name} (value) VALUES (2)"
                        )
                        raise AsyncAtomicTestError("Oops")
                self.assertTrue(async_connection.get_rollback())
                with self.assertRaisesMessage(
                    transaction.TransactionManagementError,
                    msg,
                ):
                    await async_connection.execute(f"SELECT COUNT(*) FROM {table_name}")

            row = await self._count_rows(async_connection, table_name)
            await async_connection.execute(f"DROP TABLE {table_name}")

        self.assertEqual(row, (0,))

    async def test_async_atomic_rollback_on_cancelled_error(self):
        table_name = "async_support_atomic_cancelled"
        started = asyncio.Event()

        async with await connection.new_async_connection() as async_connection:
            await self._reset_table(async_connection, table_name)

        async def worker():
            async with await connection.new_async_connection() as worker_connection:
                async with worker_connection.atomic():
                    await worker_connection.execute(
                        f"INSERT INTO {table_name} (value) VALUES (1)"
                    )
                    started.set()
                    await asyncio.sleep(60)

        task = asyncio.create_task(worker())
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        async with await connection.new_async_connection() as verifier_connection:
            row = await self._count_rows(verifier_connection, table_name)
            await verifier_connection.execute(f"DROP TABLE {table_name}")

        self.assertEqual(row, (0,))

    async def test_async_atomic_select_for_update_nowait(self):
        table_name = "async_support_atomic_for_update"
        async with await connection.new_async_connection() as async_connection:
            await self._reset_table(async_connection, table_name)
            await async_connection.execute(
                f"INSERT INTO {table_name} (value) VALUES (1)"
            )

            async with async_connection.atomic():
                async with async_connection.cursor() as cursor:
                    await cursor.execute(
                        f"SELECT id FROM {table_name} WHERE value = 1 FOR UPDATE"
                    )
                    await cursor.fetchone()

                async with await connection.new_async_connection() as other_connection:
                    with self.assertRaises(DatabaseError):
                        async with other_connection.atomic():
                            await other_connection.execute(
                                "SELECT id FROM "
                                f"{table_name} WHERE value = 1 FOR UPDATE NOWAIT"
                            )

            await async_connection.execute(f"DROP TABLE {table_name}")

    async def test_native_async_queryset_create_get_count_and_save(self):
        async with await connection.new_async_connection() as async_connection:
            queryset = CharFieldModel.objects.all().using_async_connection(
                async_connection
            )

            created = await queryset.acreate(field="alpha")
            count = await queryset.acount()
            fetched = await queryset.filter(pk=created.pk).aget()
            fetched.field = "beta"
            await fetched.asave(
                async_connection=async_connection,
                update_fields=["field"],
            )
            updated = await queryset.filter(pk=created.pk).aget()

        self.assertEqual(count, 1)
        self.assertEqual(fetched.pk, created.pk)
        self.assertEqual(updated.field, "beta")

    async def test_native_async_queryset_create_rolls_back_with_atomic(self):
        async with await connection.new_async_connection() as async_connection:
            queryset = CharFieldModel.objects.all().using_async_connection(
                async_connection
            )

            with self.assertRaisesMessage(AsyncAtomicTestError, "Oops"):
                async with async_connection.atomic():
                    await queryset.acreate(field="alpha")
                    raise AsyncAtomicTestError("Oops")

            count = await queryset.acount()

        self.assertEqual(count, 0)

    async def test_native_async_queryset_select_for_update_nowait(self):
        async with await connection.new_async_connection() as first_connection:
            first_queryset = CharFieldModel.objects.all().using_async_connection(
                first_connection
            )
            created = await first_queryset.acreate(field="alpha")

            async with first_connection.atomic():
                locked = await first_queryset.select_for_update().aget(pk=created.pk)
                self.assertEqual(locked.pk, created.pk)

                async with await connection.new_async_connection() as second_connection:
                    second_queryset = CharFieldModel.objects.all().using_async_connection(
                        second_connection
                    )
                    with self.assertRaises(DatabaseError):
                        async with second_connection.atomic():
                            await second_queryset.select_for_update(
                                nowait=True
                            ).aget(pk=created.pk)

    async def test_native_async_queryset_aexists(self):
        async with await connection.new_async_connection() as async_connection:
            queryset = CharFieldModel.objects.all().using_async_connection(
                async_connection
            )
            self.assertFalse(await queryset.aexists())

            await queryset.acreate(field="alpha")

            self.assertTrue(await queryset.aexists())
            self.assertFalse(await queryset.filter(field="missing").aexists())

    async def test_native_async_queryset_afirst_and_alast(self):
        async with await connection.new_async_connection() as async_connection:
            queryset = CharFieldModel.objects.all().using_async_connection(
                async_connection
            )

            self.assertIsNone(await queryset.afirst())
            self.assertIsNone(await queryset.alast())

            first_created = await queryset.acreate(field="zeta")
            second_created = await queryset.acreate(field="alpha")

            first_obj = await queryset.afirst()
            last_obj = await queryset.alast()

        self.assertEqual(first_obj.pk, first_created.pk)
        self.assertEqual(last_obj.pk, second_created.pk)

    async def test_native_async_queryset_async_for_uses_native_connection(self):
        async with await connection.new_async_connection() as async_connection:
            queryset = CharFieldModel.objects.all().using_async_connection(
                async_connection
            )
            await queryset.acreate(field="beta")
            await queryset.acreate(field="alpha")
            ordered = queryset.order_by("field")

            with unittest.mock.patch(
                "django.db.models.query.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                values = [obj.field async for obj in ordered]
                cached_values = [obj.field async for obj in ordered]

        self.assertEqual(values, ["alpha", "beta"])
        self.assertEqual(cached_values, values)

    async def test_native_async_queryset_aiterator_uses_chunked_cursor(self):
        async with await connection.new_async_connection() as async_connection:
            queryset = CharFieldModel.objects.all().using_async_connection(
                async_connection
            )
            await queryset.acreate(field="alpha")
            await queryset.acreate(field="beta")
            await queryset.acreate(field="gamma")
            ordered = queryset.order_by("pk")

            with (
                unittest.mock.patch(
                    "django.db.models.query.sync_to_async",
                    side_effect=AssertionError(
                        "sync_to_async bridge should not be used"
                    ),
                ),
                unittest.mock.patch.object(
                    async_connection,
                    "chunked_cursor",
                    wraps=async_connection.chunked_cursor,
                ) as chunked_cursor,
            ):
                values = [obj.field async for obj in ordered.aiterator(chunk_size=1)]

        self.assertEqual(values, ["alpha", "beta", "gamma"])
        self.assertEqual(chunked_cursor.call_count, 1)

    async def test_native_async_values_queryset_uses_native_connection(self):
        async with await connection.new_async_connection() as async_connection:
            queryset = CharFieldModel.objects.all().using_async_connection(
                async_connection
            )
            first_created = await queryset.acreate(field="beta")
            second_created = await queryset.acreate(field="alpha")
            values_queryset = queryset.values("id", "field").order_by("field")

            with unittest.mock.patch(
                "django.db.models.query.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                values = [row async for row in values_queryset]
                cached_values = [row async for row in values_queryset]
                first_row = await values_queryset.afirst()
                last_row = await values_queryset.alast()
                fetched_row = await values_queryset.filter(field="alpha").aget()

        self.assertEqual(
            values,
            [
                {"id": second_created.pk, "field": "alpha"},
                {"id": first_created.pk, "field": "beta"},
            ],
        )
        self.assertEqual(cached_values, values)
        self.assertEqual(first_row, values[0])
        self.assertEqual(last_row, values[1])
        self.assertEqual(fetched_row, values[0])

    async def test_native_async_values_list_variants_use_native_connection(self):
        async with await connection.new_async_connection() as async_connection:
            queryset = CharFieldModel.objects.all().using_async_connection(
                async_connection
            )
            first_created = await queryset.acreate(field="beta")
            second_created = await queryset.acreate(field="alpha")
            tuple_queryset = queryset.values_list("id", "field").order_by("field")
            flat_queryset = queryset.values_list("field", flat=True).order_by("field")
            named_queryset = queryset.values_list(
                "id", "field", named=True
            ).order_by("field")

            with unittest.mock.patch(
                "django.db.models.query.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                tuple_rows = [row async for row in tuple_queryset]
                first_tuple = await tuple_queryset.afirst()
                flat_rows = [
                    value async for value in flat_queryset.aiterator(chunk_size=1)
                ]
                named_row = await named_queryset.filter(field="beta").aget()

        self.assertEqual(
            tuple_rows,
            [
                (second_created.pk, "alpha"),
                (first_created.pk, "beta"),
            ],
        )
        self.assertEqual(first_tuple, tuple_rows[0])
        self.assertEqual(flat_rows, ["alpha", "beta"])
        self.assertEqual(named_row.id, first_created.pk)
        self.assertEqual(named_row.field, "beta")

    async def test_native_async_queryset_helpers_use_native_connection(self):
        async with await connection.new_async_connection() as async_connection:
            queryset = CharFieldModel.objects.all().using_async_connection(
                async_connection
            )
            first_created = await queryset.acreate(field="beta")
            second_created = await queryset.acreate(field="alpha")

            with unittest.mock.patch(
                "django.db.models.query.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                aggregate = await queryset.aaggregate(
                    total=Count("id"),
                    max_pk=Max("pk"),
                )
                earliest = await queryset.aearliest("field")
                latest = await queryset.alatest("field")
                values_earliest = await queryset.values("field").aearliest("field")
                flat_latest = await queryset.values_list(
                    "field", flat=True
                ).alatest("field")
                contains_first = await queryset.acontains(first_created)
                contains_on_filtered = await queryset.filter(field="alpha").acontains(
                    first_created
                )

        self.assertEqual(aggregate, {"total": 2, "max_pk": second_created.pk})
        self.assertEqual(earliest.pk, second_created.pk)
        self.assertEqual(latest.pk, first_created.pk)
        self.assertEqual(values_earliest, {"field": "alpha"})
        self.assertEqual(flat_latest, "beta")
        self.assertIs(contains_first, True)
        self.assertIs(contains_on_filtered, False)

    async def test_native_async_queryset_aupdate_uses_native_connection(self):
        async with await connection.new_async_connection() as async_connection:
            queryset = CharFieldModel.objects.all().using_async_connection(
                async_connection
            )
            created = await queryset.acreate(field="alpha")

            with unittest.mock.patch(
                "django.db.models.query.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                rows = await queryset.filter(pk=created.pk).aupdate(field="beta")
                updated = await queryset.aget(pk=created.pk)

        self.assertEqual(rows, 1)
        self.assertEqual(updated.field, "beta")

    async def test_native_async_queryset_aget_or_create_uses_native_connection(self):
        async with await connection.new_async_connection() as async_connection:
            queryset = CharFieldModel.objects.all().using_async_connection(
                async_connection
            )

            with unittest.mock.patch(
                "django.db.models.query.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                created_obj, created = await queryset.aget_or_create(field="alpha")
                fetched_obj, fetched_created = await queryset.aget_or_create(
                    field="alpha"
                )

        self.assertIs(created, True)
        self.assertIs(fetched_created, False)
        self.assertEqual(fetched_obj.pk, created_obj.pk)
        self.assertEqual(fetched_obj.field, "alpha")

    async def test_native_async_queryset_aupdate_or_create_uses_native_connection(self):
        async with await connection.new_async_connection() as async_connection:
            queryset = CharFieldModel.objects.all().using_async_connection(
                async_connection
            )
            existing = await queryset.acreate(field="alpha")

            with unittest.mock.patch(
                "django.db.models.query.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                updated_obj, updated_created = await queryset.aupdate_or_create(
                    pk=existing.pk,
                    defaults={"field": "beta"},
                )
                created_obj, created = await queryset.aupdate_or_create(
                    field="gamma",
                    defaults={"field": "gamma"},
                )

        self.assertIs(updated_created, False)
        self.assertEqual(updated_obj.pk, existing.pk)
        self.assertEqual(updated_obj.field, "beta")
        self.assertIs(created, True)
        self.assertEqual(created_obj.field, "gamma")

    async def test_native_async_queryset_abulk_create_uses_native_connection(self):
        async with await connection.new_async_connection() as async_connection:
            queryset = CharFieldModel.objects.all().using_async_connection(
                async_connection
            )
            objs = [
                CharFieldModel(field="alpha"),
                CharFieldModel(field="beta"),
            ]

            with unittest.mock.patch(
                "django.db.models.query.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                created = await queryset.abulk_create(objs)
                values = [obj.field async for obj in queryset.order_by("field")]

        self.assertEqual(created, objs)
        self.assertEqual(values, ["alpha", "beta"])
        self.assertTrue(all(obj.pk is not None for obj in created))
        self.assertTrue(all(obj._state.adding is False for obj in created))
        self.assertTrue(all(obj._state.db == "default" for obj in created))

    async def test_native_async_queryset_abulk_update_uses_native_connection(self):
        async with await connection.new_async_connection() as async_connection:
            queryset = CharFieldModel.objects.all().using_async_connection(
                async_connection
            )
            objs = await queryset.abulk_create(
                [
                    CharFieldModel(field="alpha"),
                    CharFieldModel(field="beta"),
                ]
            )
            objs[0].field = "gamma"
            objs[1].field = "delta"

            with unittest.mock.patch(
                "django.db.models.query.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                rows = await queryset.abulk_update(objs, ["field"])
                values = [obj.field async for obj in queryset.order_by("field")]

        self.assertEqual(rows, 2)
        self.assertEqual(values, ["delta", "gamma"])

    async def test_native_async_queryset_ain_bulk_uses_native_connection(self):
        async with await connection.new_async_connection() as async_connection:
            queryset = CharFieldModel.objects.all().using_async_connection(
                async_connection
            )
            created = await queryset.abulk_create(
                [
                    CharFieldModel(field="alpha"),
                    CharFieldModel(field="beta"),
                ]
            )

            with unittest.mock.patch(
                "django.db.models.query.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                result = await queryset.ain_bulk([created[1].pk, created[0].pk])

        self.assertEqual(list(result), [created[0].pk, created[1].pk])
        self.assertEqual(result[created[0].pk].field, "alpha")
        self.assertEqual(result[created[1].pk].field, "beta")

    async def test_native_async_queryset_ain_bulk_values_shapes_use_native_connection(
        self,
    ):
        async with await connection.new_async_connection() as async_connection:
            queryset = CharFieldModel.objects.all().using_async_connection(
                async_connection
            )
            created = await queryset.abulk_create(
                [
                    CharFieldModel(field="alpha"),
                    CharFieldModel(field="beta"),
                ]
            )

            with unittest.mock.patch(
                "django.db.models.query.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                values_result = await queryset.values("field").ain_bulk(
                    [created[0].pk, created[1].pk]
                )
                values_list_result = await queryset.values_list("field").ain_bulk(
                    [created[0].pk, created[1].pk]
                )
                named_result = await queryset.values_list("field", named=True).ain_bulk(
                    [created[0].pk, created[1].pk]
                )
                flat_result = await queryset.values_list("field", flat=True).ain_bulk(
                    [created[0].pk, created[1].pk]
                )

        self.assertEqual(
            values_result,
            {
                created[0].pk: {"field": "alpha"},
                created[1].pk: {"field": "beta"},
            },
        )
        self.assertEqual(
            values_list_result,
            {
                created[0].pk: ("alpha",),
                created[1].pk: ("beta",),
            },
        )
        self.assertEqual(named_result[created[0].pk]._fields, ("pk", "field"))
        self.assertEqual(named_result[created[0].pk].pk, created[0].pk)
        self.assertEqual(named_result[created[0].pk].field, "alpha")
        self.assertEqual(named_result[created[1].pk]._fields, ("pk", "field"))
        self.assertEqual(named_result[created[1].pk].pk, created[1].pk)
        self.assertEqual(named_result[created[1].pk].field, "beta")
        self.assertEqual(
            flat_result,
            {
                created[0].pk: "alpha",
                created[1].pk: "beta",
            },
        )

    async def test_native_async_queryset_prefetch_related_aget_uses_native_connection(
        self,
    ):
        async with await connection.new_async_connection() as async_connection:
            scene_queryset = Scene.objects.all().using_async_connection(async_connection)
            character_queryset = Character.objects.all().using_async_connection(
                async_connection
            )
            queryset = Line.objects.all().using_async_connection(async_connection)
            scene = await scene_queryset.acreate(scene="Intro", setting="Castle")
            character = await character_queryset.acreate(name="Arthur")
            line = await queryset.acreate(
                scene=scene,
                character=character,
                dialogue="Bring out your dead.",
            )

            with unittest.mock.patch(
                "django.db.models.query.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                fetched = await queryset.prefetch_related("scene", "character").aget(
                    pk=line.pk
                )

        self.assertTrue(Line._meta.get_field("scene").is_cached(fetched))
        self.assertTrue(Line._meta.get_field("character").is_cached(fetched))
        self.assertEqual(fetched.scene.setting, "Castle")
        self.assertEqual(fetched.character.name, "Arthur")

    async def test_native_async_queryset_prefetch_related_prefetch_object_uses_native_connection(
        self,
    ):
        async with await connection.new_async_connection() as async_connection:
            scene_queryset = Scene.objects.all().using_async_connection(async_connection)
            character_queryset = Character.objects.all().using_async_connection(
                async_connection
            )
            queryset = Line.objects.all().using_async_connection(async_connection)
            first_scene = await scene_queryset.acreate(
                scene="Intro",
                setting="Castle",
            )
            second_scene = await scene_queryset.acreate(
                scene="Outro",
                setting="Village",
            )
            character = await character_queryset.acreate(name="Patsy")
            await queryset.abulk_create(
                [
                    Line(
                        scene=second_scene,
                        character=character,
                        dialogue="It is but a scratch.",
                    ),
                    Line(
                        scene=first_scene,
                        character=character,
                        dialogue="Ni!",
                    ),
                ]
            )

            with unittest.mock.patch(
                "django.db.models.query.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                fetched = await queryset.order_by("dialogue").prefetch_related(
                    Prefetch(
                        "scene",
                        queryset=Scene.objects.filter(setting="Castle"),
                        to_attr="prefetched_scene",
                    )
                ).afirst()

        self.assertEqual(fetched.dialogue, "It is but a scratch.")
        self.assertTrue(hasattr(fetched, "prefetched_scene"))
        self.assertIsNone(fetched.prefetched_scene)

    async def test_native_async_instance_asave_uses_state_connection(self):
        async with await connection.new_async_connection() as async_connection:
            queryset = CharFieldModel.objects.all().using_async_connection(
                async_connection
            )
            obj = await queryset.acreate(field="alpha")

            with unittest.mock.patch(
                "django.db.models.base.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                obj.field = "beta"
                await obj.asave()
                refreshed = await queryset.aget(pk=obj.pk)

        self.assertEqual(refreshed.field, "beta")

    async def test_native_async_instance_arefresh_from_db_uses_state_connection(self):
        async with await connection.new_async_connection() as async_connection:
            queryset = CharFieldModel.objects.all().using_async_connection(
                async_connection
            )
            obj = await queryset.acreate(field="alpha")
            await async_connection.execute(
                "UPDATE postgres_tests_charfieldmodel SET field = %s WHERE id = %s",
                ["gamma", obj.pk],
            )

            with unittest.mock.patch(
                "django.db.models.base.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                await obj.arefresh_from_db()

        self.assertEqual(obj.field, "gamma")

    async def test_native_async_related_manager_reads_use_instance_connection(self):
        async with await connection.new_async_connection() as async_connection:
            scene_queryset = Scene.objects.all().using_async_connection(async_connection)
            character_queryset = Character.objects.all().using_async_connection(
                async_connection
            )
            line_queryset = Line.objects.all().using_async_connection(async_connection)
            scene = await scene_queryset.acreate(scene="Bridge", setting="Bridge")
            character = await character_queryset.acreate(name="Bedevere")
            await line_queryset.abulk_create(
                [
                    Line(scene=scene, character=character, dialogue="First"),
                    Line(scene=scene, character=character, dialogue="Second"),
                ]
            )

            with unittest.mock.patch(
                "django.db.models.query.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                count = await scene.line_set.acount()
                first = await scene.line_set.order_by("dialogue").afirst()

        self.assertEqual(count, 2)
        self.assertEqual(first.dialogue, "First")

    async def test_native_async_select_related_instances_keep_connection_state(self):
        async with await connection.new_async_connection() as async_connection:
            scene_queryset = Scene.objects.all().using_async_connection(async_connection)
            character_queryset = Character.objects.all().using_async_connection(
                async_connection
            )
            queryset = Line.objects.all().using_async_connection(async_connection)
            scene = await scene_queryset.acreate(scene="Intro", setting="Castle")
            character = await character_queryset.acreate(name="Arthur")
            line = await queryset.acreate(
                scene=scene,
                character=character,
                dialogue="Bring out your dead.",
            )
            fetched = await queryset.select_related("scene").aget(pk=line.pk)

            with unittest.mock.patch(
                "django.db.models.base.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                fetched.scene.setting = "Swamp"
                await fetched.scene.asave()
                await scene.arefresh_from_db()

        self.assertEqual(scene.setting, "Swamp")

    async def test_native_async_reverse_related_manager_acreate_uses_native_connection(
        self,
    ):
        async with await connection.new_async_connection() as async_connection:
            scene = await Scene.objects.all().using_async_connection(async_connection).acreate(
                scene="Bridge",
                setting="Bridge",
            )
            character = await Character.objects.all().using_async_connection(
                async_connection
            ).acreate(name="Robin")

            with (
                unittest.mock.patch(
                    "django.db.models.fields.related_descriptors.sync_to_async",
                    side_effect=AssertionError(
                        "sync_to_async bridge should not be used"
                    ),
                ),
                unittest.mock.patch(
                    "django.db.models.query.sync_to_async",
                    side_effect=AssertionError(
                        "sync_to_async bridge should not be used"
                    ),
                ),
            ):
                created = await scene.line_set.acreate(
                    character=character,
                    dialogue="Brave Sir Robin.",
                )
                count = await scene.line_set.acount()

        self.assertEqual(created.scene_id, scene.pk)
        self.assertEqual(created.dialogue, "Brave Sir Robin.")
        self.assertEqual(count, 1)

    async def test_native_async_reverse_related_manager_aadd_uses_native_connection(
        self,
    ):
        async with await connection.new_async_connection() as async_connection:
            scene_queryset = Scene.objects.all().using_async_connection(async_connection)
            character_queryset = Character.objects.all().using_async_connection(
                async_connection
            )
            line_queryset = Line.objects.all().using_async_connection(async_connection)
            source_scene = await scene_queryset.acreate(
                scene="Source",
                setting="Forest",
            )
            target_scene = await scene_queryset.acreate(
                scene="Target",
                setting="Castle",
            )
            character = await character_queryset.acreate(name="Lancelot")
            line = await line_queryset.acreate(
                scene=source_scene,
                character=character,
                dialogue="Charge!",
            )

            with (
                unittest.mock.patch(
                    "django.db.models.fields.related_descriptors.sync_to_async",
                    side_effect=AssertionError(
                        "sync_to_async bridge should not be used"
                    ),
                ),
                unittest.mock.patch(
                    "django.db.models.base.sync_to_async",
                    side_effect=AssertionError(
                        "sync_to_async bridge should not be used"
                    ),
                ),
            ):
                await target_scene.line_set.aadd(line)
                await line.arefresh_from_db()

        self.assertEqual(line.scene_id, target_scene.pk)

    async def test_native_async_reverse_related_manager_aget_or_create_uses_native_connection(
        self,
    ):
        async with await connection.new_async_connection() as async_connection:
            scene = await Scene.objects.all().using_async_connection(async_connection).acreate(
                scene="Hill",
                setting="Hill",
            )
            character = await Character.objects.all().using_async_connection(
                async_connection
            ).acreate(name="Galahad")

            with (
                unittest.mock.patch(
                    "django.db.models.fields.related_descriptors.sync_to_async",
                    side_effect=AssertionError(
                        "sync_to_async bridge should not be used"
                    ),
                ),
                unittest.mock.patch(
                    "django.db.models.query.sync_to_async",
                    side_effect=AssertionError(
                        "sync_to_async bridge should not be used"
                    ),
                ),
            ):
                created_obj, created = await scene.line_set.aget_or_create(
                    dialogue="Let me have just a little bit of peril?",
                    defaults={"character": character},
                )
                fetched_obj, fetched = await scene.line_set.aget_or_create(
                    dialogue="Let me have just a little bit of peril?",
                    defaults={"character": character},
                )

        self.assertIs(created, True)
        self.assertIs(fetched, False)
        self.assertEqual(created_obj.pk, fetched_obj.pk)
        self.assertEqual(created_obj.scene_id, scene.pk)

    async def test_native_async_reverse_related_manager_aupdate_or_create_uses_native_connection(
        self,
    ):
        async with await connection.new_async_connection() as async_connection:
            scene = await Scene.objects.all().using_async_connection(async_connection).acreate(
                scene="Castle",
                setting="Castle",
            )
            character_queryset = Character.objects.all().using_async_connection(
                async_connection
            )
            first_character = await character_queryset.acreate(name="Arthur")
            second_character = await character_queryset.acreate(name="Bedevere")
            existing = await scene.line_set.acreate(
                character=first_character,
                dialogue="Old dialogue",
            )

            with (
                unittest.mock.patch(
                    "django.db.models.fields.related_descriptors.sync_to_async",
                    side_effect=AssertionError(
                        "sync_to_async bridge should not be used"
                    ),
                ),
                unittest.mock.patch(
                    "django.db.models.query.sync_to_async",
                    side_effect=AssertionError(
                        "sync_to_async bridge should not be used"
                    ),
                ),
            ):
                updated_obj, updated_created = await scene.line_set.aupdate_or_create(
                    pk=existing.pk,
                    defaults={
                        "character": second_character,
                        "dialogue": "New dialogue",
                    },
                )
                created_obj, created = await scene.line_set.aupdate_or_create(
                    dialogue="Fresh dialogue",
                    defaults={"character": first_character},
                )

        self.assertIs(updated_created, False)
        self.assertEqual(updated_obj.pk, existing.pk)
        self.assertEqual(updated_obj.dialogue, "New dialogue")
        self.assertEqual(updated_obj.character_id, second_character.pk)
        self.assertIs(created, True)
        self.assertEqual(created_obj.scene_id, scene.pk)

    async def test_native_async_queryset_prefetch_related_reverse_fk_uses_native_connection(
        self,
    ):
        async with await connection.new_async_connection() as async_connection:
            scene_queryset = Scene.objects.all().using_async_connection(async_connection)
            character_queryset = Character.objects.all().using_async_connection(
                async_connection
            )
            line_queryset = Line.objects.all().using_async_connection(async_connection)
            scene = await scene_queryset.acreate(scene="Camp", setting="Camp")
            character = await character_queryset.acreate(name="Tim")
            await line_queryset.abulk_create(
                [
                    Line(scene=scene, character=character, dialogue="Alpha"),
                    Line(scene=scene, character=character, dialogue="Beta"),
                ]
            )

            with unittest.mock.patch(
                "django.db.models.query.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                fetched = await scene_queryset.prefetch_related("line_set").aget(
                    pk=scene.pk
                )

        cached_queryset = fetched._prefetched_objects_cache["line_set"]
        self.assertEqual(
            sorted(line.dialogue for line in cached_queryset),
            ["Alpha", "Beta"],
        )

    async def test_native_async_queryset_prefetch_related_reverse_fk_to_attr_uses_native_connection(
        self,
    ):
        async with await connection.new_async_connection() as async_connection:
            scene_queryset = Scene.objects.all().using_async_connection(async_connection)
            character_queryset = Character.objects.all().using_async_connection(
                async_connection
            )
            line_queryset = Line.objects.all().using_async_connection(async_connection)
            scene = await scene_queryset.acreate(scene="Tower", setting="Tower")
            character = await character_queryset.acreate(name="Concorde")
            await line_queryset.abulk_create(
                [
                    Line(scene=scene, character=character, dialogue="Alpha"),
                    Line(scene=scene, character=character, dialogue="Beta"),
                ]
            )

            with unittest.mock.patch(
                "django.db.models.query.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                fetched = await scene_queryset.prefetch_related(
                    Prefetch(
                        "line_set",
                        queryset=Line.objects.filter(dialogue__startswith="B"),
                        to_attr="filtered_lines",
                    )
                ).aget(pk=scene.pk)

        self.assertEqual([line.dialogue for line in fetched.filtered_lines], ["Beta"])

    async def test_native_async_many_to_many_manager_reads_use_instance_connection(
        self,
    ):
        async with await connection.new_async_connection() as async_connection:
            scene_queryset = Scene.objects.all().using_async_connection(async_connection)
            character_queryset = Character.objects.all().using_async_connection(
                async_connection
            )
            scene = await scene_queryset.acreate(scene="Bridge", setting="Bridge")
            characters = await character_queryset.abulk_create(
                [
                    Character(name="Arthur"),
                    Character(name="Lancelot"),
                ]
            )
            await scene.characters.aadd(*characters)

            with (
                unittest.mock.patch(
                    "django.db.models.fields.related_descriptors.sync_to_async",
                    side_effect=AssertionError(
                        "sync_to_async bridge should not be used"
                    ),
                ),
                unittest.mock.patch(
                    "django.db.models.query.sync_to_async",
                    side_effect=AssertionError(
                        "sync_to_async bridge should not be used"
                    ),
                ),
            ):
                count = await scene.characters.acount()
                names = [
                    character.name
                    async for character in scene.characters.order_by("name")
                ]

        self.assertEqual(count, 2)
        self.assertEqual(names, ["Arthur", "Lancelot"])

    async def test_native_async_many_to_many_manager_write_helpers_use_native_connection(
        self,
    ):
        async with await connection.new_async_connection() as async_connection:
            scene_queryset = Scene.objects.all().using_async_connection(async_connection)
            character_queryset = Character.objects.all().using_async_connection(
                async_connection
            )
            scene = await scene_queryset.acreate(scene="Castle", setting="Castle")
            keep, remove, add_later = await character_queryset.abulk_create(
                [
                    Character(name="Keep"),
                    Character(name="Remove"),
                    Character(name="Add later"),
                ]
            )

            with (
                unittest.mock.patch(
                    "django.db.models.fields.related_descriptors.sync_to_async",
                    side_effect=AssertionError(
                        "sync_to_async bridge should not be used"
                    ),
                ),
                unittest.mock.patch(
                    "django.db.models.query.sync_to_async",
                    side_effect=AssertionError(
                        "sync_to_async bridge should not be used"
                    ),
                ),
            ):
                await scene.characters.aadd(keep, remove)
                await scene.characters.aremove(remove)
                await scene.characters.aset([keep, add_later])
                after_set = [
                    character.name
                    async for character in scene.characters.order_by("name")
                ]
                await scene.characters.aclear()
                final_count = await scene.characters.acount()

        self.assertEqual(after_set, ["Add later", "Keep"])
        self.assertEqual(final_count, 0)

    async def test_native_async_many_to_many_manager_object_creation_uses_native_connection(
        self,
    ):
        async with await connection.new_async_connection() as async_connection:
            scene_queryset = Scene.objects.all().using_async_connection(async_connection)
            scene = await scene_queryset.acreate(scene="Grail", setting="Cave")

            with (
                unittest.mock.patch(
                    "django.db.models.fields.related_descriptors.sync_to_async",
                    side_effect=AssertionError(
                        "sync_to_async bridge should not be used"
                    ),
                ),
                unittest.mock.patch(
                    "django.db.models.query.sync_to_async",
                    side_effect=AssertionError(
                        "sync_to_async bridge should not be used"
                    ),
                ),
            ):
                created = await scene.characters.acreate(name="Tim")
                fetched, created_flag = await scene.characters.aget_or_create(
                    name="Patsy"
                )
                updated, updated_created = await scene.characters.aupdate_or_create(
                    name="Dennis",
                    defaults={},
                )
                related_names = [
                    character.name
                    async for character in scene.characters.order_by("name")
                ]

        self.assertEqual(created.name, "Tim")
        self.assertEqual(fetched.name, "Patsy")
        self.assertIs(created_flag, True)
        self.assertEqual(updated.name, "Dennis")
        self.assertIs(updated_created, True)
        self.assertEqual(related_names, ["Dennis", "Patsy", "Tim"])

    async def test_native_async_queryset_prefetch_related_many_to_many_uses_native_connection(
        self,
    ):
        async with await connection.new_async_connection() as async_connection:
            scene_queryset = Scene.objects.all().using_async_connection(async_connection)
            character_queryset = Character.objects.all().using_async_connection(
                async_connection
            )
            scene = await scene_queryset.acreate(scene="Shrubbery", setting="Forest")
            characters = await character_queryset.abulk_create(
                [
                    Character(name="Knight"),
                    Character(name="Roger"),
                ]
            )
            await scene.characters.aadd(*characters)

            with unittest.mock.patch(
                "django.db.models.query.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                fetched = await scene_queryset.prefetch_related("characters").aget(
                    pk=scene.pk
                )

        cached_queryset = fetched._prefetched_objects_cache["characters"]
        self.assertEqual(
            sorted(character.name for character in cached_queryset),
            ["Knight", "Roger"],
        )

    async def test_native_async_queryset_prefetch_related_many_to_many_to_attr_uses_native_connection(
        self,
    ):
        async with await connection.new_async_connection() as async_connection:
            scene_queryset = Scene.objects.all().using_async_connection(async_connection)
            character_queryset = Character.objects.all().using_async_connection(
                async_connection
            )
            scene = await scene_queryset.acreate(scene="Cave", setting="Cave")
            characters = await character_queryset.abulk_create(
                [
                    Character(name="Black Beast"),
                    Character(name="Bors"),
                    Character(name="Tim"),
                ]
            )
            await scene.characters.aadd(*characters)

            with unittest.mock.patch(
                "django.db.models.query.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                fetched = await scene_queryset.prefetch_related(
                    Prefetch(
                        "characters",
                        queryset=Character.objects.filter(name__startswith="B"),
                        to_attr="b_characters",
                    )
                ).aget(pk=scene.pk)

        self.assertEqual(
            sorted(character.name for character in fetched.b_characters),
            ["Black Beast", "Bors"],
        )

    async def test_native_async_queryset_prefetch_related_reverse_many_to_many_uses_native_connection(
        self,
    ):
        async with await connection.new_async_connection() as async_connection:
            scene_queryset = Scene.objects.all().using_async_connection(async_connection)
            character_queryset = Character.objects.all().using_async_connection(
                async_connection
            )
            first_scene = await scene_queryset.acreate(scene="Hill", setting="Hill")
            second_scene = await scene_queryset.acreate(scene="Moat", setting="Moat")
            character = await character_queryset.acreate(name="Robin")
            await first_scene.characters.aadd(character)
            await second_scene.characters.aadd(character)

            with unittest.mock.patch(
                "django.db.models.query.sync_to_async",
                side_effect=AssertionError("sync_to_async bridge should not be used"),
            ):
                fetched = await character_queryset.prefetch_related("scenes").aget(
                    pk=character.pk
                )

        cached_queryset = fetched._prefetched_objects_cache["scenes"]
        self.assertEqual(
            sorted(scene.scene for scene in cached_queryset),
            ["Hill", "Moat"],
        )
