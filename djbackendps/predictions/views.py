from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from .models import PredictionRecord
from .serializers import PredictionSerializer

class PredictionViewSet(viewsets.ModelViewSet):
    queryset = PredictionRecord.objects.all()
    serializer_class = PredictionSerializer

    def create(self, request, *args, **kwargs):
        # Extract data from request
        data = request.data
        prediction_type = data.get('prediction_type')
        prediction_data = data.get('prediction_data', {})
        prediction_result = data.get('prediction_result', '')
        risk_score = data.get('risk_score', 0.0)

        # Create a record based on prediction type
        record_data = {
            'disease_type': prediction_type,
            'prediction_result': 'Has' in prediction_result,  # Convert to boolean
            'risk_score': risk_score,
            # Don't include username field to avoid database schema issues
            'age': prediction_data.get('age'),
            'gender': prediction_data.get('gender'),
        }

        # Add type-specific fields
        if prediction_type == 'diabetes':
            record_data.update({
                'pregnancies': prediction_data.get('pregnancies'),
                'glucose': prediction_data.get('glucose'),
                'blood_pressure': prediction_data.get('blood_pressure'),
                'skin_thickness': prediction_data.get('skin_thickness'),
                'insulin': prediction_data.get('insulin'),
                'bmi': prediction_data.get('bmi'),
                'diabetes_pedigree': prediction_data.get('pedigree'),
            })
        elif prediction_type == 'heart':
            record_data.update({
                'chest_pain_type': prediction_data.get('cp'),
                'resting_bp': prediction_data.get('trestbps'),
                'cholesterol': prediction_data.get('chol'),
                'fasting_blood_sugar': prediction_data.get('fbs') == 'Yes',
                'rest_ecg': prediction_data.get('restecg'),
                'max_heart_rate': prediction_data.get('thalach'),
                'exercise_induced_angina': prediction_data.get('exang') == 'Yes',
                'st_depression': prediction_data.get('oldpeak'),
                'st_slope': prediction_data.get('slope'),
                'num_major_vessels': prediction_data.get('ca'),
                'thalassemia': prediction_data.get('thal'),
            })
        elif prediction_type == 'kidney':
            # Add kidney-specific fields when implemented
            pass

        # Create and save the record
        serializer = self.get_serializer(data=record_data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    # Add custom actions for each prediction type
    @action(detail=False, methods=['post'], url_path='save')
    def save_prediction(self, request):
        print(f"Received prediction data: {request.data}")
        try:
            response = self.create(request)
            print(f"Created prediction with response: {response.data}")
            return response
        except Exception as e:
            print(f"Error creating prediction: {e}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
