import unittest

from django.db import connection
from django.test import TransactionTestCase, modify_settings

try:
    from django.db.backends.postgresql.psycopg_any import is_psycopg3
except ImportError:
    is_psycopg3 = False


@unittest.skipUnless(connection.vendor == "postgresql", "PostgreSQL specific tests")
@unittest.skipUnless(is_psycopg3, "Native async PostgreSQL support requires psycopg 3")
@modify_settings(INSTALLED_APPS={"append": "django.contrib.postgres"})
class PostgreSQLAsyncSupportTests(TransactionTestCase):
    available_apps = ["django.contrib.postgres"]

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
