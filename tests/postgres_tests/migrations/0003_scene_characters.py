from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("postgres_tests", "0002_create_test_models"),
    ]

    operations = [
        migrations.AddField(
            model_name="scene",
            name="characters",
            field=models.ManyToManyField(
                blank=True,
                related_name="scenes",
                to="postgres_tests.character",
            ),
        ),
    ]
