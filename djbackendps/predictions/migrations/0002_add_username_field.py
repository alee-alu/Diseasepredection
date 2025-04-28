from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictions', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='predictionrecord',
            name='username',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
    ]
