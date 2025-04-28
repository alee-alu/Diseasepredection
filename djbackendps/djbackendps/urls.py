"""
URL configuration for djbackendps project.
"""
import datetime
from django.contrib import admin
from django.urls import path
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Import the include function from django.urls
from django.urls import include

# Simple health check endpoint
@csrf_exempt
def health_check(request):
    print(f"Health check requested from {request.META.get('REMOTE_ADDR')}")
    return JsonResponse({'status': 'ok', 'message': 'API is healthy', 'timestamp': str(datetime.datetime.now())}, status=200)

urlpatterns = [
    path('admin/', admin.site.urls),
    # Health check endpoint
    path('api/health/', health_check, name='health_check'),
    # Test endpoint
    path('test/', lambda request: JsonResponse({'message': 'Test endpoint works!'}), name='test'),
    # Include the predictions URLs
    path('api/', include('predictions.urls')),
]
