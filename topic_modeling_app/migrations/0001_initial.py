# Generated by Django 4.1.13 on 2024-01-05 14:58

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='LdaModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('num_topics', models.IntegerField()),
                ('chunksize', models.IntegerField()),
                ('decay', models.FloatField()),
                ('gamma_threshold', models.FloatField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='LsaModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('num_topics', models.IntegerField()),
                ('chunksize', models.IntegerField()),
                ('decay', models.FloatField()),
            ],
        ),
        migrations.CreateModel(
            name='PertinentWords',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('word', models.CharField(max_length=50)),
                ('frequency', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='PlsaModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('num_topics', models.IntegerField()),
                ('passes', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='UserSelection',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('preprocessing_option', models.CharField(max_length=100)),
                ('encodation_option', models.CharField(max_length=100)),
                ('model_option', models.CharField(max_length=100)),
            ],
        ),
    ]
