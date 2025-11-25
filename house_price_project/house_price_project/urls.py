from django.contrib import admin
from django.urls import path
from predictor.views import home, predict_house_price

urlpatterns = [
    path('', home, name='home'),                      # 👈 homepage
    path('admin/', admin.site.urls),
    path('api/predict/', predict_house_price, name='predict'),
]
