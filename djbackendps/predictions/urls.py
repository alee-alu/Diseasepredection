from django.urls import path, include
from django.http import JsonResponse
from rest_framework.routers import DefaultRouter
from rest_framework.decorators import api_view
from . import views

# Health check endpoint
@api_view(['GET'])
def health_check(request):
    # Return a JSON response with status 200
    return JsonResponse({'status': 'ok', 'message': 'API is healthy'}, status=200)

# Create a router for the viewset
router = DefaultRouter()
router.register(r'predictions', views.PredictionViewSet)

# Original URL patterns
urlpatterns = [
    path('health/', health_check, name='health_check'),
    path('diabetes/', views.PredictionViewSet.as_view({'post': 'create'}), name='save_diabetes_prediction'),
    path('heart/', views.PredictionViewSet.as_view({'post': 'create'}), name='save_heart_prediction'),
    path('kidney/', views.PredictionViewSet.as_view({'post': 'create'}), name='save_kidney_prediction'),
]

# Add the router URLs to the urlpatterns
urlpatterns += router.urls
