# Generated by Django 5.0 on 2025-03-21 06:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0013_alter_userprofile_roadmap'),
    ]

    operations = [
        migrations.AddField(
            model_name='userprofile',
            name='projects',
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
