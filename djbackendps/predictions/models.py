# === File: prediction/models.py ===
from django.db import models

class PredictionRecord(models.Model):
    DISEASE_CHOICES = [
        ('diabetes', 'Diabetes'),
        ('heart', 'Heart Disease'),
        ('kidney', 'Kidney Disease'),
    ]

    disease_type = models.CharField(max_length=20, choices=DISEASE_CHOICES)
    prediction_result = models.BooleanField()
    risk_score = models.FloatField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    # Common fields
    username = models.CharField(max_length=50, null=True, blank=True)
    age = models.IntegerField(null=True, blank=True)
    gender = models.CharField(max_length=10, null=True, blank=True)

    # Diabetes-specific fields
    pregnancies = models.IntegerField(null=True, blank=True)
    glucose = models.FloatField(null=True, blank=True)
    blood_pressure = models.FloatField(null=True, blank=True)
    skin_thickness = models.FloatField(null=True, blank=True)
    insulin = models.FloatField(null=True, blank=True)
    bmi = models.FloatField(null=True, blank=True)
    diabetes_pedigree = models.FloatField(null=True, blank=True)

    # Heart-specific fields
    chest_pain_type = models.IntegerField(null=True, blank=True)
    resting_bp = models.FloatField(null=True, blank=True)
    cholesterol = models.FloatField(null=True, blank=True)
    fasting_blood_sugar = models.BooleanField(null=True, blank=True)
    rest_ecg = models.IntegerField(null=True, blank=True)
    max_heart_rate = models.FloatField(null=True, blank=True)
    exercise_induced_angina = models.BooleanField(null=True, blank=True)
    st_depression = models.FloatField(null=True, blank=True)
    st_slope = models.IntegerField(null=True, blank=True)
    num_major_vessels = models.IntegerField(null=True, blank=True)
    thalassemia = models.IntegerField(null=True, blank=True)

    # Kidney-specific fields
    blood_urea = models.FloatField(null=True, blank=True)
    blood_glucose_random = models.FloatField(null=True, blank=True)
    creatinine = models.FloatField(null=True, blank=True)
    albumin = models.FloatField(null=True, blank=True)
    sodium = models.FloatField(null=True, blank=True)
    potassium = models.FloatField(null=True, blank=True)
    hemoglobin = models.FloatField(null=True, blank=True)
    packed_cell_volume = models.FloatField(null=True, blank=True)
    white_blood_cell_count = models.FloatField(null=True, blank=True)
    red_blood_cell_count = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.disease_type} prediction on {self.timestamp}"
