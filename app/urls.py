from django.urls import path
from . import views
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import LoginUserView, MyTokenObtainPairView, CVUploadView
from .views import (
    register_user,
   check_authentication,
    fetch_profile, get_career_transition_plan, google_auth ,google_auth1, get_career_roadmap , get_career_projects, get_user_profile 
)




urlpatterns = [

    path('api/token/', MyTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('register/', register_user, name='register_user'),
     path('auth/google/', google_auth, name='google-auth'),
     path('api/auth/google/', google_auth1, name='google_auth1'),

    path('api/login/', LoginUserView.as_view(), name='login_user'),
    
  
    path('user/profile/', get_user_profile, name='user-profile'),

     path('update-password-after-reset/', views.update_password_after_reset),

    # path('reset-password/<str:token>/', views.reset_password, name='reset_password'),
    path('check-auth/', check_authentication, name='check-auth'),
    path('profile/', views.save_profile, name='save_profile'),
    path('fetch-profile/', fetch_profile, name='fetch_profile'),
    path('upload-cv/', CVUploadView.as_view(), name='upload-cv'), 
    path("roadmap/", get_career_roadmap, name="get_career_roadmap"),
    path("projects/", get_career_projects, name="get_career_projects"),
     path("career-transition/", get_career_transition_plan, name="career-transition"),
     
#resume analyzer
     path('upload-resume/', views.upload_resume, name='upload_resume'),

#match with job description
    path('match-resumes/', views.match_resumes, name='match_resumes'),
    path('resume-matcher/', views.match_resumes, name='resume_matcher'),

 
]


