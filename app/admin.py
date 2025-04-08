# admin.py
from django.contrib import admin
from .models import UserData, UserProfile, CareerRoadmap

admin.site.register(UserData)
admin.site.register(UserProfile)
admin.site.register(CareerRoadmap)
# admin.site.register(CV)
