from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.hashers import make_password
from django.core.mail import send_mail
from django.conf import settings
from .models import UserData
import json

@csrf_exempt
def register_user(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            # Extract data
            name = data.get('name')
            email = data.get('email')
            password = data.get('password')
            firebase_uid = data.get('firebase_uid')  # New field from frontend

            # Validate input
            if not all([name, email, password]):
                return JsonResponse({'error': 'All fields are required!'}, status=400)

            # Check if email is already registered
            if UserData.objects.filter(email=email).exists():
                return JsonResponse({'error': 'Email already registered!'}, status=400)

            # For Firebase users, check if UID exists
            if firebase_uid and UserData.objects.filter(firebase_uid=firebase_uid).exists():
                return JsonResponse({'error': 'Firebase user already registered!'}, status=400)

            # Hash the password before storing (only for email/password users)
            hashed_password = make_password(password) if password else None

            # Save data in UserData model
            user = UserData.objects.create(
                name=name,
                email=email,
                password=hashed_password,
                firebase_uid=firebase_uid,  # Store Firebase UID
                is_verified=bool(firebase_uid)  # Auto-verify Firebase users
            )

            
            if not firebase_uid:
                user.generate_token()
               

            return JsonResponse({
                'message': 'User registered successfully!',
                'verification_required': not bool(firebase_uid)  
            }, status=201)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data!'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method!'}, status=405)


@csrf_exempt
def google_auth(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Extract Google user data
            email = data.get('email')
            name = data.get('name')
            firebase_uid = data.get('firebase_uid')
            
            # Validate required fields
            if not all([email, firebase_uid]):
                return JsonResponse({'error': 'Email and Firebase UID are required'}, status=400)
            
            try:
                user = UserData.objects.get(email=email)
                
                # Update user data if needed
                update_fields = []
                if not user.firebase_uid:
                    user.firebase_uid = firebase_uid
                    update_fields.append('firebase_uid')
                if not user.name and name:
                    user.name = name
                    update_fields.append('name')
                
                # Mark as Google auth user
                if not user.is_google_auth:
                    user.is_google_auth = True
                    update_fields.append('is_google_auth')
                if not user.is_verified:
                    user.is_verified = True
                    update_fields.append('is_verified')
                
                if update_fields:
                    user.save(update_fields=update_fields)
                
                # Generate new token
                user.generate_token()
                
                return JsonResponse({
                    'message': 'Google authentication successful',
                    'user': {
                        'name': user.name,
                        'email': user.email,
                        'token': user.token,
                        'is_google_auth': True,
                        'is_verified': True,
                        'userType': 'user'  # Explicitly set userType
                    }
                }, status=200)
                
            except UserData.DoesNotExist:
                # Create new user for Google auth
                user = UserData.objects.create(
                    name=name or email.split('@')[0],
                    email=email,
                    password=make_password(None),
                    firebase_uid=firebase_uid,
                    is_google_auth=True,
                    is_verified=True,
                    # usertype='user'  # Also set in model if you have this field
                )
                
                user.generate_token()
                
                return JsonResponse({
                    'message': 'New Google user created successfully',
                    'user': {
                        'name': user.name,
                        'email': user.email,
                        'token': user.token,
                        'is_google_auth': True,
                        'is_verified': True,
                        'userType': 'user'  # Explicitly set userType
                    }
                }, status=201)
                
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)


@csrf_exempt  # Only if you're not using CSRF token in your frontend
def google_auth1(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Extract Google user data
            email = data.get('email')
            firebase_uid = data.get('firebase_uid')
            
            # Validate required fields
            if not all([email, firebase_uid]):
                return JsonResponse({'error': 'Email and Firebase UID are required'}, status=400)
            
            try:
                user = UserData.objects.get(email=email)
                
                # Check if email is verified
                if not user.is_verified:
                    return JsonResponse({'error': 'Email not verified. Please verify your email first.'}, status=403)
                
                # Update only the firebase_uid if it's not set
                update_fields = []
                if not user.firebase_uid:
                    user.firebase_uid = firebase_uid
                    update_fields.append('firebase_uid')
                
                # Mark as Google auth user if not already
                if not user.is_google_auth:
                    user.is_google_auth = True
                    update_fields.append('is_google_auth')
                
                if update_fields:
                    user.save(update_fields=update_fields)
                
                # Generate new token
                user.generate_token()
                
                return JsonResponse({
                    'message': 'Google authentication successful',
                    'user': {
                        'name': user.name,  # Keep existing name
                        'email': user.email,
                        'token': user.token,
                        'is_google_auth': True,
                        'is_verified': True,
                        'userType': "user"
                    }
                }, status=200)
                
            except UserData.DoesNotExist:
                # Don't create new user, return error
                return JsonResponse({
                    'error': 'No account found with this email. Please sign up first.'
                }, status=404)
                
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)


    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            email = data.get('email')
            new_password = data.get('new_password')
            firebase_uid = data.get('firebase_uid')  # Added for Firebase integration
            
            if not all([email, new_password]):
                return JsonResponse({'error': 'Email and new password are required'}, status=400)
            
            try:
                user = UserData.objects.get(email=email)
                
                # Additional check for Firebase users
                if user.firebase_uid and firebase_uid and user.firebase_uid != firebase_uid:
                    return JsonResponse({'error': 'Invalid user credentials'}, status=403)
                
                # Update password in Django database
                user.password = make_password(new_password)
                user.save()
                
                return JsonResponse({'message': 'Password updated successfully'}, status=200)
            
            except UserData.DoesNotExist:
                return JsonResponse({'error': 'No user found with this email'}, status=404)
                
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)


@csrf_exempt
def update_password_after_reset(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            email = data.get("email")
            password = data.get("password")
            firebase_uid = data.get("firebase_uid")

            user = UserData.objects.get(email=email, firebase_uid=firebase_uid)

            user.password = make_password(password)  # hash it!
            user.save()

            return JsonResponse({"message": "Password updated in backend"}, status=200)
        except UserData.DoesNotExist:
            return JsonResponse({"error": "User not found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token['email'] = user.email  # Add custom claims
        return token

# Register the custom serializer
from rest_framework_simplejwt.views import TokenObtainPairView
class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer

from django.contrib.auth.models import User
from .models import UserData
from django.contrib.auth.hashers import check_password
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.status import HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_401_UNAUTHORIZED

class LoginUserView(APIView):
    def post(self, request):
        data = request.data
        identifier = data.get('identifier')
        password = data.get('password')

        if not identifier or not password:
            return Response({'error': 'Username/email and password are required!'}, status=HTTP_400_BAD_REQUEST)
        
    

        try:
            user = None
            user_type = None

            # Check for admin in User model
            try:
                user = User.objects.get(username=identifier)
                user_type = "admin"
            except User.DoesNotExist:
                # Check for user in UserData model
                user = UserData.objects.get(email=identifier)
                user_type = "user"
                if not user.is_verified:  # Assume `user.is_verified` checks if the email is verified
                    verification_link = f"{settings.FRONTEND_URL}/verify-email"  # You can set this in your Django settings
                    return Response({
                        'error': 'Please verify your email. A verification email has been sent.',
                        'verification_link': verification_link
                    }, status=HTTP_400_BAD_REQUEST)

            if not check_password(password, user.password):
                return Response({'error': 'Invalid credentials!'}, status=HTTP_401_UNAUTHORIZED)

            # Generate JWT Token
            refresh = RefreshToken.for_user(user)
            access = refresh.access_token

            return Response({
                'message': 'Login successful!',
                'refresh': str(refresh),
                'access': str(access),
                'username': user.username if hasattr(user, 'username') else user.name,
                'userType': user_type,
            }, status=HTTP_200_OK)

        except (UserData.DoesNotExist, User.DoesNotExist):
            return Response({'error': 'Invalid username/email or password!'}, status=HTTP_401_UNAUTHORIZED)




from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import UserProfile

@api_view(['GET'])
def get_user_profile(request):
    try:
        # Get the first user profile (or modify as needed)
        profile = UserProfile.objects.first()  # Or any other non-auth query
        return Response({
            'work_preference': profile.work_preference,
            'employment_type': profile.employment_type,
            'expected_salary': profile.expected_salary
        })
    except UserProfile.DoesNotExist:
        return Response({'error': 'No profile found'}, status=404)
    
 
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import UserData
import json
from django.contrib.auth.hashers import make_password

@csrf_exempt
def forgot_password(request):
    """Endpoint to check if email exists in database before sending reset link"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            email = data.get('email')
            
            if not email:
                return JsonResponse({'error': 'Email is required!'}, status=400)
                
            exists = UserData.objects.filter(email=email).exists()
            return JsonResponse({'exists': exists}, status=200)
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data!'}, status=400)
    
    return JsonResponse({'error': 'Invalid request method!'}, status=405)


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import UserData
from django.contrib.auth.hashers import make_password, check_password
import json

@csrf_exempt
def update_django_password(request):
    print("\n=== Password Update Request ===")
    
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    try:
        # Parse request
        data = json.loads(request.body)
        email = data.get('email', '').strip()
        new_password = data.get('password', '').strip()
        
        print(f"Email: {email}, Password: {'*' * len(new_password)}")
        
        # Validate
        if not email or not new_password:
            raise ValueError("Email and password required")
        if len(new_password) < 8:
            raise ValueError("Password must be 8+ characters")
        
        # Get user
        user = UserData.objects.get(email=email)
        print(f"Found user: {user.email}")
        
        # Hash and save password
        user.set_password(new_password)  # Using model method
        user.save()
        
        # Verify
        user.refresh_from_db()
        if not user.check_password(new_password):
            raise Exception("Password verification failed")
            
        print("Password updated successfully")
        return JsonResponse({'success': True, 'message': 'Password updated'})
        
    except UserData.DoesNotExist:
        print("User not found")
        return JsonResponse({'error': 'User not found'}, status=404)
    except Exception as e:
        print(f"Error: {str(e)}")
        return JsonResponse({'error': str(e)}, status=400)


#forgot
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.core.mail import send_mail
# from django.conf import settings
# from .models import UserData, PasswordResetToken
# import json
# from django.utils.timezone import now
# from datetime import timedelta
# import uuid

# @csrf_exempt
# def forgot_password(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             email = data.get('email')

#             if not email:
#                 return JsonResponse({'error': 'Email is required!'}, status=400)

#             try:
#                 user = UserData.objects.get(email=email)

#                 # Generate password reset token
#                 token = uuid.uuid4().hex
#                 expiration = now() + timedelta(hours=1)  # Token valid for 1 hour
#                 PasswordResetToken.objects.create(user=user, token=token, expired_at=expiration)

#                 # Send reset email
#                 send_reset_password_email(user, token)

#                 return JsonResponse({'message': 'Password reset link has been sent to your email.'}, status=200)

#             except UserData.DoesNotExist:
#                 return JsonResponse({'error': 'Email not found!'}, status=404)

#         except json.JSONDecodeError:
#             return JsonResponse({'error': 'Invalid JSON data!'}, status=400)

#     return JsonResponse({'error': 'Invalid request method!'}, status=405)


# def send_reset_password_email(user, token):
#     reset_link = f"{settings.FRONTEND_URL}/reset-password/{token}"

#   # Ensure this is set in your settings.py
    

#     subject = "Reset Your Password"
#     message = f"Hi {user.name},\n\nClick the link below to reset your password:\n{reset_link}\n\nThis link will expire in 1 hour.\n\nThank you!"

#     send_mail(subject, message, settings.DEFAULT_FROM_EMAIL, [user.email], fail_silently=False)


# #reset view
# from django.contrib.auth.hashers import make_password

# @csrf_exempt
# def reset_password(request, token):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             new_password = data.get('password')

#             if not new_password:
#                 return JsonResponse({'error': 'Password is required!'}, status=400)

#             try:
#                 reset_token = PasswordResetToken.objects.get(token=token)

#                 if reset_token.expired_at > now():
#                     user = reset_token.user
#                     user.password = make_password(new_password)  # Hash the new password
#                     user.save()

#                     # Delete token after successful reset
#                     reset_token.delete()

#                     return JsonResponse({'message': 'Password reset successfully!'}, status=200)
#                 else:
#                     return JsonResponse({'error': 'Reset link has expired!'}, status=400)

#             except PasswordResetToken.DoesNotExist:
#                 return JsonResponse({'error': 'Invalid reset link!'}, status=404)

#         except json.JSONDecodeError:
#             return JsonResponse({'error': 'Invalid JSON data!'}, status=400)

#     return JsonResponse({'error': 'Invalid request method!'}, status=405)





from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework.exceptions import AuthenticationFailed
from django.contrib.auth.models import User

@api_view(['GET'])
def check_authentication(request):
    # Check if the user is authenticated using JWT
    if request.user.is_authenticated:
        return Response({'is_authenticated': True, 'username': request.user.username})
    else:
        raise AuthenticationFailed('Authentication failed')





from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import requests
import os
import joblib
import numpy as np
from scipy.sparse import hstack
from .models import UserData, UserProfile

# Load model and preprocessing tools
from django.conf import settings

MODEL_DIR = os.path.join(settings.BASE_DIR, 'models')  # Relative to the Django project

model = joblib.load(os.path.join(MODEL_DIR, 'career_recommendation_model.pkl'))
tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))


MISTRAL_API_KEY = "gahK0sTPMF0HOhbmFOj3nl7D865xJB3e"  # Replace with actual API key

# ✅ Function to validate recommendations with Mistral AI
def validate_recommendations_with_ai(recommendations, skills):
    try:
        print("🔍 Validating recommendations with AI...")

        api_url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
        }
        payload = {
            "model": "mistral-small",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI expert in career guidance. "
                               "Validate the following career recommendations based on the given skills. "
                               "If any recommended field is outdated, replace it with a new, suitable, and high-trending career. "
                               "Ensure the response is a valid JSON list of dictionaries with 'career' and 'probability' fields only. "
                               "DO NOT return any extra explanations or text—ONLY a JSON array."
                },
                {
                    "role": "user",
                    "content": f"Skills: {skills}. Current career recommendations: {recommendations}. "
                               "Return a JSON array only, like this: "
                               "[{{'career': 'Software Engineer', 'probability': 80.0}}, {{'career': 'AI Specialist', 'probability': 70.0}}]"
                }
            ],
            "temperature": 0.5,
        }

        response = requests.post(api_url, headers=headers, json=payload)
        data = response.json()

        print("🔍 AI Response:", data)  # Debugging

        if "choices" in data and len(data["choices"]) > 0:
            ai_response_content = data["choices"][0]["message"]["content"]

            try:
                updated_recommendations = json.loads(ai_response_content)

                # Ensure it's a valid list of dictionaries
                if isinstance(updated_recommendations, list) and all(isinstance(item, dict) for item in updated_recommendations):
                    print('✅ AI validation completed!')
                    return updated_recommendations
                else:
                    print("⚠️ AI Response is not in expected format. Using original recommendations.")
                    return recommendations  # Fallback

            except json.JSONDecodeError:
                print("⚠️ AI Response is not valid JSON. Using original recommendations.")
                return recommendations  # Return original if parsing fails

    except Exception as e:
        print(f"⚠️ AI Validation Error: {e}")
        return recommendations  # Return original if API fails

# ✅ Generate Recommendations (Make sure this function exists)
def generate_recommendations(skills, experience, education, interests, job_demand):
    print("🚀 Generating recommendations...")
    print("📌 Skills:", skills)
    print("📌 Experience:", experience)
    print("📌 Education:", education)
    print("📌 Interests:", interests)
    print("📌 Job Market Demand Score:", job_demand)

    # Example: Dummy recommendations (Replace with ML model logic)
    recommendations = [
        {"career": "Software Engineer", "probability": 70.0},
        {"career": "Data Scientist", "probability": 60.0},
        {"career": "AI Specialist", "probability": 50.0}
    ]
    
    print("✅ Initial Recommendations:", recommendations)
    return recommendations

@csrf_exempt
def save_profile(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            print("📩 Received Data:", data)  # Debugging output

            username = data.get('username')
            if not username:
                return JsonResponse({'error': 'Username is required!'}, status=400)

            try:
                user = UserData.objects.get(name=username)
            except UserData.DoesNotExist:
                return JsonResponse({'error': 'User not found!'}, status=404)

            profile, created = UserProfile.objects.get_or_create(user=user)

            # ✅ Update user profile fields
            profile.first_name = data.get('firstName', profile.first_name)
            profile.last_name = data.get('lastName', profile.last_name)
            profile.dob = data.get('dob', profile.dob)
            profile.gender = data.get('gender', profile.gender)

            country_data = data.get('country', '')
            profile.country = country_data.get("value", "") if isinstance(country_data, dict) else country_data
            profile.city = data.get('city', profile.city)
            profile.phone_number = data.get('phone', profile.phone_number)

            # ✅ Skills & Career Data
            skills = data.get('skills', profile.skills)
            profile.skills = skills  # Save in profile
            profile.experience = data.get('experienceLevel', profile.experience)
            profile.education = data.get('education', profile.education)
            profile.personal_interests = data.get('personalInterests', profile.personal_interests)

            job_market_demand = data.get('job_market_demand_score', profile.job_market_demand_score)
            profile.job_market_demand_score = int(job_market_demand) if job_market_demand else 0

            # ✅ Handle career transformation fields
            career_transform = data.get('careerTransform', 'no')
            profile.career_transform = career_transform

            if career_transform == 'yes':
                profile.previous_job = data.get('previousJob', '') or None
                total_experience = data.get('totalExperience', 0)
                profile.total_experience = int(total_experience) if str(total_experience).isdigit() else None
                profile.skill_level = data.get('skillLevel', '') or None
                profile.current_skills = data.get('currentSkills', '') or None
                profile.career_field = data.get('careerField', '') or None
                profile.future_goal = data.get('futureGoal', '') or None
                profile.employment_type = data.get('employmentType', '') or None
                profile.work_preference = data.get('workPreference', '') or None
                profile.expected_salary = data.get('expectedSalary', '') or None
            else:
                profile.previous_job = None
                profile.total_experience = None
                profile.skill_level = None
                profile.current_skills = None
                profile.career_field = None
                profile.future_goal = None
                profile.employment_type = None
                profile.work_preference = None
                profile.expected_salary = None

            profile.save()  # ✅ Save before generating recommendations

            # ✅ Generate recommendations
            recommendations = generate_recommendations(
                skills=skills,
                experience=profile.experience,  
                education=profile.education,
                interests=profile.personal_interests,
                job_demand=profile.job_market_demand_score
            )

            # 🔄 Validate recommendations with AI
            validated_recommendations = validate_recommendations_with_ai(recommendations, skills)

            # ✅ Save updated recommendations
            profile.recommendations = json.dumps(validated_recommendations)  # Ensure proper JSON format
            profile.save()
            print("✅ Final Recommendations:", validated_recommendations)  # Debugging output

            return JsonResponse({
                'message': 'Profile saved and recommendations updated!',
                'recommendations': validated_recommendations
            }, status=200)

        except ValueError as ve:
            print("🚨 ValueError:", str(ve))
            return JsonResponse({'error': f'Invalid data format: {str(ve)}'}, status=400)

        except Exception as e:
            print("🚨 Unexpected Error:", str(e))
            return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request method!'}, status=405)



     



def generate_recommendations(skills, experience, education, interests, job_demand):
    try:
        # Preprocess inputs, defaulting to neutral values if any field is missing
        skills_vectorized = tfidf.transform([skills if skills else ""])
        experience_encoded = label_encoders["Experience Level"].transform([experience if experience else "Entry"])
        education_encoded = label_encoders["Education Level"].transform([education if education else "High School"])
        interests_encoded = label_encoders["Personal Interests"].transform([interests if interests else "General"])
        job_demand_scaled = scaler.transform([[float(job_demand) if job_demand else 0]])

        # Combine features into a single input
        input_data = hstack([
            skills_vectorized,
            [[experience_encoded[0], education_encoded[0], interests_encoded[0], job_demand_scaled[0][0]]]
        ])

        # Predict probabilities
        probabilities = model.predict_proba(input_data)[0]
        top_indices = np.argsort(probabilities)[-3:][::-1]

        # Get top 3 recommendations
        recommendation_le = label_encoders["Corrected Recommendation"]
        recommendations = [
            {'career': recommendation_le.inverse_transform([idx])[0], 'probability': round(probabilities[idx] * 100, 2)}
            for idx in top_indices
        ]

        return recommendations

    except Exception as e:
        # Log the error clearly for debugging purposes
        print(f"Error during recommendation generation: {str(e)}")
        return [{'error': f'Error generating recommendations: {str(e)}'}]




#fetch profile 
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import UserData, UserProfile
import json

# Fetch user profile info
from django.http import JsonResponse
from .models import UserData, UserProfile

from django.http import JsonResponse
from .models import UserData, UserProfile

import json

import json

def fetch_profile(request):
    if request.method == 'GET':
        username = request.GET.get('username')

        if not username:
            return JsonResponse({'error': 'Username not provided!'}, status=400)

        try:
            user = UserData.objects.get(name=username)
            profile = UserProfile.objects.get(user=user)

            # ✅ Extract only the country name
            country_name = profile.country
            if country_name and country_name.startswith("{"):
                try:
                    country_dict = json.loads(country_name)  # Convert JSON string to dictionary
                    country_name = country_dict.get("value", country_name)  # Extract country name
                except json.JSONDecodeError:
                    pass  # If decoding fails, keep the original value

            return JsonResponse({
                'username': user.name,
                'email': user.email,
                'firstName': profile.first_name,
                'lastName': profile.last_name,
                'dob': profile.dob.strftime('%Y-%m-%d') if profile.dob else None,
                'gender': profile.gender,
                 'country': profile.country or "",  # ✅ Now only country name is returned
                'city': profile.city,
                'phone': profile.phone_number,
                'skills': profile.skills or "",
                'experienceLevel': profile.experience or "",  # Match frontend naming
                'education': profile.education or "",
                'personalInterests': profile.personal_interests or "",
                'job_market_demand_score': profile.job_market_demand_score or 0,
                'recommendations': profile.recommendations or "",

                # ✅ New fields with fallback values
                'careerTransform': profile.career_transform or "",  # Match frontend
                'previousJob': profile.previous_job or "",
                'totalExperience': profile.total_experience if profile.total_experience is not None else 0,
                'skillLevel': profile.skill_level or "",
                'currentSkills' : profile.current_skills or "",
                'careerField': profile.career_field or "",
                'futureGoal': profile.future_goal or "",
                'employmentType': profile.employment_type or "",
                'workPreference': profile.work_preference or "",
                'expectedSalary': profile.expected_salary or "",

            }, status=200)

        except (UserData.DoesNotExist, UserProfile.DoesNotExist):
            return JsonResponse({'error': 'Profile not found!'}, status=404)

    return JsonResponse({'error': 'Invalid request method!'}, status=405)


import os
import json
import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import get_object_or_404
from .models import UserData, UserProfile  # Ensure correct import

# Secure API key handling
MISTRAL_API_KEY = "gahK0sTPMF0HOhbmFOj3nl7D865xJB3e"

@csrf_exempt
def get_career_transition_plan(request):
    """API Endpoint to generate and retrieve a structured career transition roadmap."""

    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method. Please use POST!"}, status=400)

    try:
        # ✅ Parse and validate request data
        data = json.loads(request.body)
        required_fields = ["username", "previous_job", "career_field"]
        
        missing_fields = [field for field in required_fields if not data.get(field, "").strip()]
        if missing_fields:
            return JsonResponse({"error": f"Missing required fields: {', '.join(missing_fields)}"}, status=400)

        # ✅ Extract data
        username = data.get("username")
        previous_job = data.get("previous_job")
        career_field = data.get("career_field")
        total_experience = data.get("total_experience", "N/A")
        skill_level = data.get("skill_level", "N/A")
        current_skills = data.get("current_skills", "N/A")
        future_goal = data.get("future_goal", "N/A")
        employment_type = data.get("employment_type", "N/A")
        work_preference = data.get("work_preference", "N/A")
        expected_salary = data.get("expected_salary", "N/A")

        # ✅ Check if user exists
        user = get_object_or_404(UserData, name=username)
        profile, created = UserProfile.objects.get_or_create(user=user)

        # ✅ Regenerate plan every time user updates the profile
        generate_new_plan = True

        # ✅ Call Mistral AI API
        api_url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
        }

        # Structured AI Prompt
        prompt_content = (
     "Generate a structured career transition plan in clean HTML format."
    "Use `<strong>` for section titles and `<ul><li>` for bullet points."
    "Ensure the response is **strictly formatted as valid HTML** with no markdown or special symbols.\n\n"

    f"<strong>Career Transition Plan</strong><br><br>"

    f"<strong>Overview</strong><br>"
    f"<ul>"
    f"<li><strong>Previous Job:</strong> {previous_job}</li>"
    f"<li><strong>Target Career Field:</strong> {career_field}</li>"
    f"<li><strong>Experience Level:</strong> {total_experience} years</li>"
    f"<li><strong>Current Skills:</strong> {current_skills}</li>"
    f"<li><strong>Future Goal:</strong> {future_goal}</li>"
    f"<li><strong>Preferred Employment Type:</strong> {employment_type}</li>"
    f"<li><strong>Work Preference:</strong> {work_preference}</li>"
    f"<li><strong>Expected Salary:</strong> ${expected_salary}</li>"
    f"</ul><br>"

    f"<strong>Key Skills to Learn</strong><br>"
    f"<ul>"
    f"<li>Identify and list the most in-demand skills in {career_field}.</li>"
    f"<li>Suggest technical, soft, and transferable skills needed for the transition.</li>"
    f"</ul><br>"

    f"<strong>Recommended Certifications & Courses</strong><br>"
    f"<ul>"
    f"<li>List professional certifications and online courses for {career_field}.</li>"
    f"</ul><br>"

    f"<strong>Job Search Strategies</strong><br>"
    f"<ul>"
    f"<li>How to network and build connections in {career_field}.</li>"
    f"<li>Resume optimization and cover letter best practices.</li>"
    f"</ul><br>"

    f"<strong>Estimated Transition Timeline</strong><br>"
    f"<ul>"
    f"<li>Suggested timeframe for skill development and job applications.</li>"
    f"</ul><br>"

    f"<strong>Final Career Advice</strong><br>"
    f"<ul>"
    f"<li>Industry-specific tips and growth opportunities.</li>"
    f"</ul><br>"

    "Ensure the response follows valid HTML formatting without missing any sections."
)



        payload = {
            "model": "mistral-small",
            "messages": [
                {"role": "system", "content": "You are an AI career coach providing structured, professional career transition plans."},
                {"role": "user", "content": prompt_content},
            ],
            "temperature": 0.7,
        }

        response = requests.post(api_url, headers=headers, json=payload)
        response_data = response.json()

        # ✅ Validate AI Response
        transition_plan = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not transition_plan:
            return JsonResponse({"error": "Failed to generate career transition plan."}, status=500)

        # ✅ Save transition plan to database
        profile.previous_job = previous_job
        profile.career_field = career_field
        profile.current_skills = current_skills
        profile.expected_salary = expected_salary
        profile.transition_plan = transition_plan
        profile.save()

        return JsonResponse({"transition_plan": transition_plan}, status=200)

    except UserData.DoesNotExist:
        return JsonResponse({"error": "User not found!"}, status=404)

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON format."}, status=400)

    except requests.RequestException as e:
        return JsonResponse({"error": f"External API error: {str(e)}"}, status=500)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)






from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status
import PyPDF2
import re
import spacy
import yake  # Keyword extraction
import nltk
from nltk.tokenize import sent_tokenize
from word2number import w2n

nlp = spacy.load("en_core_web_sm")
nltk.download("punkt")

# YAKE Skill Extraction
def extract_skills(text):
    kw_extractor = yake.KeywordExtractor(top=15)
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

# Validate Email
def validate_email(email):
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(pattern, email) is not None

# Validate Phone Number
def validate_phone(phone):
    pattern = r'^\+?\d{1,3}?[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}$'
    return re.match(pattern, phone) is not None

# Extract Contact Info
def extract_contact_info(text):
    emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    email = next((e for e in emails if validate_email(e)), None)

    phones = re.findall(r'\+?\d[\d -]{8,15}\d', text)
    phone = next((p for p in phones if validate_phone(p)), None)

    return email, phone

# Extract Name
def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

# Extract Experience
def extract_experience(text):
    experience_keywords = ["years", "year", "experience", "entry", "mid", "senior"]
    
    for line in text.split('\n'):
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in experience_keywords):
            num_years = re.findall(r'(\d+)\s*(?:\+*\s*years?)', line_lower)
            if num_years:
                return f"{num_years[0]} years experience"

            words = line_lower.split()
            for i in range(len(words)):
                try:
                    num = w2n.word_to_num(words[i])
                    if i + 1 < len(words) and words[i + 1] in ["years", "year"]:
                        return f"{num} years experience"
                except ValueError:
                    continue

    return "Not Found"

# Extract CV Title (Job Title)
def extract_cv_title(text):
    doc = nlp(text)
    job_title = None

    for ent in doc.ents:
        if ent.label_ in ["ORG", "WORK_OF_ART"]:
            job_title = ent.text
            break

    if not job_title:
        job_title_patterns = [
            r"(?:working as|worked as|currently a|currently working as)\s+([\w\s-]+)",
            r"(?:position|role|designation):\s*([\w\s-]+)"
        ]
        for pattern in job_title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                job_title = match.group(1)
                break

    return job_title if job_title else "Not Found"

# ✅ **NEW FEATURE: ATS Score Calculation**
def calculate_ats_score(text, job_description):
    """
    Calculates ATS score based on keyword match and resume structure.
    """
    job_keywords = extract_skills(job_description) if job_description else []
    resume_keywords = extract_skills(text)
    
    keyword_match = len(set(job_keywords) & set(resume_keywords)) / max(len(job_keywords), 1) * 100  # Match %
    
    # Structure Check (penalty if missing sections)
    sections = ["summary", "experience", "skills", "education"]
    section_count = sum(1 for sec in sections if sec in text.lower())
    structure_score = (section_count / len(sections)) * 100  

    # Formatting Check (checking bullet points, readability)
    bullet_points = text.count("•") + text.count("- ")
    readability_score = min(bullet_points * 5, 100)  

    # Overall ATS score
    ats_score = (0.5 * keyword_match) + (0.3 * structure_score) + (0.2 * readability_score)
    return round(ats_score, 2)
import spacy
import yake
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Mistral API Credentials
MISTRAL_API_KEY = "gahK0sTPMF0HOhbmFOj3nl7D865xJB3e"  
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

def extract_skills(text):
    """
    Extract technical skills dynamically without using predefined lists.
    """
    doc = nlp(text.lower())

    # Extract noun phrases (potential skills)
    noun_phrases = {chunk.text for chunk in doc.noun_chunks}

    # Extract named entities related to technology, software, or skills
    named_entities = {ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "ORG", "SKILL"]}

    # Use YAKE to extract key phrases
    keyword_extractor = yake.KeywordExtractor(n=2, dedupLim=0.9, top=15)  
    yake_keywords = {kw[0] for kw in keyword_extractor.extract_keywords(text)}

    # Use TF-IDF to find relevant words
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    tfidf_terms = set(tfidf_vectorizer.get_feature_names_out())


    extracted_terms = noun_phrases.union(named_entities).union(yake_keywords).union(tfidf_terms)

   


    return list(extracted_terms)











class CVUploadView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        file = request.FILES.get('file')
        job_description = request.POST.get("job_description", "").strip() 
        if not file:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() or '' for page in pdf_reader.pages)

            # Extract information
            name = extract_name(text)
            email, phone = extract_contact_info(text)
            skills = extract_skills(text)
            experience = extract_experience(text)
            cv_title = extract_cv_title(text)

            job_description = request.data.get("job_description", "")
            print(name)
            # Calculate ATS Score
            ats_score = calculate_ats_score(text, job_description) if job_description else "No Job Description Provided"
            print('ats score',ats_score)

            # 🔹 Skill Gap Analysis
            # missing_skills = calculate_skill_gap(skills, job_description)
            # print('Missong Skills',missing_skills)

            # # 🔹 AI-Based Resume Rewriting
            # improvement_suggestions = resume_improvement_suggestions(text)
            # # print('Suggestions',improvement_suggestions)

            # # 🔹 Job Priority Scoring
            # job_list = request.data.get("job_list", [])
            # ranked_jobs = rank_job(job_list, text)

            # print('Ranked Jobs',ranked_jobs)

            return Response({
                "name": name if name else "Not Found",
                "email": email if email else "Invalid or Not Found",
                "phone": phone if phone else "Invalid or Not Found",
                "cv_title": cv_title,
                "skills": skills if skills else "Not Found",
                "experience": experience if experience else "Not Found",
                # "ats_score": ats_score,
                # "missing_skills": missing_skills,  # ✅ Show missing skills
               
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

import requests
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import UserProfile, UserData  # Ensure correct import of models

MISTRAL_API_KEY = "gahK0sTPMF0HOhbmFOj3nl7D865xJB3e"  # Replace with your actual API key

@csrf_exempt
def get_career_roadmap(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            username = data.get('username')
            career_name = data.get('careerName')

            if not username or not career_name:
                return JsonResponse({"error": "Username and career name are required!"}, status=400)

            # Fetch user profile based on username
            user = UserData.objects.get(name=username)
            profile, created = UserProfile.objects.get_or_create(user=user)

            # Check if roadmap exists
            if profile.career_name == career_name and profile.roadmap:
                return JsonResponse({"roadmap": profile.roadmap}, status=200)

            # If no roadmap, fetch from Mistral AI API
            api_url = "https://api.mistral.ai/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
            }
            payload = {
    "model": "mistral-small",
    "messages": [
        {
            "role": "system",
            "content": (
                "You are an AI that provides structured career roadmaps in valid HTML format."
                "Use `<strong>` for section titles, `<ul><li>` for bullet points, and clickable `<a>` links."
                "Ensure all responses strictly follow HTML formatting with no Markdown or special characters.\n\n"
                
                "Additionally, list essential skills in a numbered format, provide explanations, "
                "and include valid learning resource links with `class='text-teal-500 font-semibold'` for highlighting."
            )
        },
        {
            "role": "user",
            "content": (
                f"Provide a structured roadmap for becoming a {career_name}."
                "Format the response in clean HTML with the following sections:\n\n"
                
                "<strong>Career Roadmap for {career_name}</strong><br><br>"

                "<strong>1. Core Skills (Beginner to Expert)</strong><br>"
                "<ul>"

                "<li><strong>Beginner Level:</strong><br>"
                "<ul>"
                "<li><strong>Fundamentals:</strong> Learn the basics of {career_name}, its importance, and foundational concepts.</li>"
                "<li><strong>Essential Tools:</strong> Get familiar with beginner-friendly tools and technologies.</li>"
                "<li><strong>Basic Hands-On Projects:</strong> Start working on small projects to build confidence.</li>"
                "</ul></li><br>"

                "<li><strong>Intermediate Level:</strong><br>"
                "<ul>"
                "<li><strong>Deepen Knowledge:</strong> Learn advanced concepts and best practices.</li>"
                "<li><strong>Expand Skillset:</strong> Work on more complex tools and technologies relevant to {career_name}.</li>"
                "<li><strong>Real-World Applications:</strong> Apply skills to solve real-world problems and start contributing to open-source projects.</li>"
                "</ul></li><br>"

                "<li><strong>Expert Level:</strong><br>"
                "<ul>"
                "<li><strong>Master Advanced Concepts:</strong> Gain expertise in the most complex areas of {career_name}.</li>"
                "<li><strong>Leadership & Mentoring:</strong> Start teaching and mentoring others in the field.</li>"
                "<li><strong>Industry Recognition:</strong> Get certifications, speak at conferences, and build a strong professional presence.</li>"
                "</ul></li><br>"

                "</ul><br>"

                "<strong>2. Learning Resources</strong><br>"
                "<ul>"
                "<li><a href='URL' class='text-teal-500 font-semibold'>Beginner Course - Platform</a>: Covers fundamental concepts.</li>"
                "<li><a href='URL' class='text-teal-500 font-semibold'>Intermediate Course - Platform</a>: Focuses on real-world applications.</li>"
                "<li><a href='URL' class='text-teal-500 font-semibold'>Expert Course - Platform</a>: Advanced topics and mastery-level learning.</li>"
                "</ul><br>"

                "<strong>3. Practical Experience</strong><br>"
                "<ul>"
                "<li>Work on real-world projects, freelance, or contribute to open-source.</li>"
                "<li>Build a portfolio showcasing hands-on experience in {career_name}.</li>"
                "</ul><br>"

                "<strong>4. Job Search Strategies</strong><br>"
                "<ul>"
                "<li>Optimize your resume and LinkedIn profile to highlight key skills.</li>"
                "<li>Network with industry professionals, attend events, and join online communities.</li>"
                "</ul><br>"

                "<strong>5. Final Tips</strong><br>"
                "<ul>"
                "<li>Stay updated with industry trends and continuously improve your skills.</li>"
                "<li>Join {career_name} communities and participate in discussions to enhance learning.</li>"
                "</ul><br>"

                "Ensure the response follows valid HTML formatting."
            )
        },
    ],
    "temperature": 0.7,
}




            response = requests.post(api_url, headers=headers, json=payload)
            data = response.json()

            if "choices" in data and len(data["choices"]) > 0:
                roadmap_data = data["choices"][0]["message"]["content"].split("\n")

                # Update the user profile with roadmap
                profile.career_name = career_name
                profile.roadmap = roadmap_data
                profile.save()

                return JsonResponse({"roadmap": roadmap_data}, status=200)

            return JsonResponse({"error": "Failed to generate roadmap"}, status=500)

        except UserData.DoesNotExist:
            return JsonResponse({"error": "User not found!"}, status=404)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method!"}, status=400)


import json
import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import UserProfile, UserData  

MISTRAL_API_KEY = "gahK0sTPMF0HOhbmFOj3nl7D865xJB3e"

@csrf_exempt
def get_career_projects(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            username = data.get("username")
            career_name = data.get("careerName")
            refresh = data.get("refresh", False)  

            if not username or not career_name:
                return JsonResponse({"error": "Username and career name are required!"}, status=400)

            # 🔹 Get User Profile
            user = UserData.objects.get(name=username)
            profile, created = UserProfile.objects.get_or_create(user=user)

            # 🔹 Ensure projects field is a valid dictionary
            projects_dict = {}
            if profile.projects:
                try:
                    projects_dict = json.loads(profile.projects)
                    if not isinstance(projects_dict, dict):
                        projects_dict = {}
                except json.JSONDecodeError:
                    projects_dict = {}

            # 🔹 If the career name changed, force refresh
            if profile.career_name != career_name:
                print(f"🔄 Career changed: Fetching new projects for {career_name}")
                refresh = True  
                profile.career_name = career_name  # Update career name

            # ✅ Return cached projects if refresh is False
            if not refresh and career_name in projects_dict:
                print(f"✅ Returning cached projects for {career_name}")
                return JsonResponse({"projects": projects_dict[career_name]}, status=200)

            # 🔹 Fetch projects from Mistral API
            print(f"🔄 Fetching new projects for {career_name}")

            api_url = "https://api.mistral.ai/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
            }
            payload = {
                "model": "mistral-small",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an AI that provides structured career-based projects in valid HTML format."
                            "Ensure each project is categorized under Beginner-Friendly, Intermediate Challenge, and Expert-Level."
                            "Use `<strong>` for section titles and `<ul><li>` for listing projects."
                            "Ensure all responses strictly follow valid HTML formatting with no Markdown or special characters."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"List career-based projects for {career_name} in valid HTML format."
                    },
                ],
                "temperature": 0.7,
            }

            try:
                response = requests.post(api_url, headers=headers, json=payload)
                response.raise_for_status()  # Raise error if status code is not 200
                data = response.json()
            except requests.exceptions.RequestException as e:
                print("❌ API Request Error:", str(e))
                return JsonResponse({"error": "Failed to connect to API"}, status=500)

            # ✅ Extract project data
            if "choices" in data and len(data["choices"]) > 0:
                project_data = data["choices"][0]["message"]["content"]

                # 🔹 Save projects in the database
                projects_dict[career_name] = project_data
                profile.projects = json.dumps(projects_dict)
                profile.save()

                print(f"✅ Projects saved for: {career_name}")
                return JsonResponse({"projects": project_data}, status=200)

            return JsonResponse({"error": "Failed to generate projects from AI"}, status=500)

        except UserData.DoesNotExist:
            return JsonResponse({"error": "User not found!"}, status=404)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON received"}, status=500)
        except Exception as e:
            print("❌ Backend Error:", str(e))
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method!"}, status=400)










# views.py
import os
import re
import time
import datetime
import random
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
import io
import spacy
from .constants import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos

# Load Spacy model
nlp = spacy.load("en_core_web_sm")

def pdf_reader(file_path):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    
    with open(file_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()

    converter.close()
    fake_file_handle.close()
    return text

def course_recommender(course_list, num_recommendations=4):
    """Recommend courses from the given course list"""
    if not course_list:
        return []
    
    # Shuffle and select courses
    shuffled_courses = random.sample(course_list, min(len(course_list), num_recommendations))
    recommended_courses = []
    
    for idx, (c_name, c_link) in enumerate(shuffled_courses, 1):
        recommended_courses.append({
            'number': idx,
            'name': c_name,
            'link': c_link
        })
    
    return recommended_courses

def extract_resume_data(file_path):
    text = pdf_reader(file_path)
    doc = nlp(text)
    
    # Extract name (simple heuristic - first line with title case)
    name = ""
    for line in text.split('\n'):
        line = line.strip()
        if line and line.istitle() and len(line.split()) >= 2:
            name = line
            break
    
    # Extract email
    email = ""
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    if email_match:
        email = email_match.group(0)
    
    # Extract phone
    phone = ""
    phone_match = re.search(r'(\+?\d{1,3}[-\.\s]?)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}', text)
    if phone_match:
        phone = phone_match.group(0)
    
    # Extract skills (simple keyword matching)
    skills = []
    skill_keywords = [
        'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'swift', 
        'kotlin', 'go', 'r', 'matlab', 'sql', 'html', 'css', 'react', 'angular',
        'vue', 'django', 'flask', 'node', 'express', 'spring', 'hibernate',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
        'machine learning', 'deep learning', 'data science', 'data analysis',
        'artificial intelligence', 'ai', 'nlp', 'natural language processing',
        'computer vision', 'big data', 'hadoop', 'spark', 'aws', 'azure',
        'google cloud', 'docker', 'kubernetes', 'devops', 'ci/cd', 'git',
        'linux', 'unix', 'bash', 'shell scripting', 'agile', 'scrum'
    ]
    
    for token in doc:
        if token.text.lower() in skill_keywords and token.text not in skills:
            skills.append(token.text)
    
    # Count pages (simple approximation)
    with open(file_path, 'rb') as f:
        page_count = len(list(PDFPage.get_pages(f)))
    
    return {
        'name': name,
        'email': email,
        'mobile_number': phone,
        'skills': skills,
        'no_of_pages': page_count
    }

def analyze_resume(file_path):
    resume_data = extract_resume_data(file_path)
    if not resume_data:
        return None
    
    resume_text = pdf_reader(file_path)
    
    # Determine candidate level based on pages
    no_of_pages = resume_data.get('no_of_pages', 1)
    if no_of_pages == 1:
        cand_level = "Fresher"
    elif no_of_pages == 2:
        cand_level = "Intermediate"
    else:
        cand_level = "Experienced"
    
    # Skill recommendations
    ds_keyword = ['tensorflow', 'keras', 'pytorch', 'machine learning', 'deep Learning', 'flask', 'streamlit']
    web_keyword = ['react', 'django', 'node jS', 'react js', 'php', 'laravel', 'magento', 'wordpress', 
                   'javascript', 'angular js', 'c#', 'flask']
    android_keyword = ['android', 'android development', 'flutter', 'kotlin', 'xml', 'kivy']
    ios_keyword = ['ios', 'ios development', 'swift', 'cocoa', 'cocoa touch', 'xcode']
    uiux_keyword = ['ux', 'adobe xd', 'figma', 'zeplin', 'balsamiq', 'ui', 'prototyping', 'wireframes', 
                    'storyframes', 'adobe photoshop', 'photoshop', 'editing', 'adobe illustrator', 
                    'illustrator', 'adobe after effects', 'after effects', 'adobe premier pro', 
                    'premier pro', 'adobe indesign', 'indesign', 'wireframe', 'solid', 'grasp', 
                    'user research', 'user experience']

    recommended_skills = []
    reco_field = ''
    rec_course = []
    
    for skill in resume_data.get('skills', []):
        skill_lower = skill.lower()
        if skill_lower in ds_keyword:
            reco_field = 'Data Science'
            recommended_skills = ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling',
                                'Data Mining', 'Clustering & Classification', 'Data Analytics',
                                'Quantitative Analysis', 'Web Scraping', 'ML Algorithms', 'Keras',
                                'Pytorch', 'Probability', 'Scikit-learn', 'Tensorflow', "Flask",
                                'Streamlit']
            rec_course = course_recommender(ds_course, 4)
            break
        elif skill_lower in web_keyword:
            reco_field = 'Web Development'
            recommended_skills = ['React', 'Django', 'Node JS', 'React JS', 'php', 'laravel', 'Magento',
                                'wordpress', 'Javascript', 'Angular JS', 'c#', 'Flask', 'SDK']
            rec_course = course_recommender(web_course, 4)
            break
        elif skill_lower in android_keyword:
            reco_field = 'Android Development'
            recommended_skills = ['Android', 'Android development', 'Flutter', 'Kotlin', 'XML', 'Java',
                                'Kivy', 'GIT', 'SDK', 'SQLite']
            rec_course = course_recommender(android_course, 4)
            break
        elif skill_lower in ios_keyword:
            reco_field = 'IOS Development'
            recommended_skills = ['IOS', 'IOS Development', 'Swift', 'Cocoa', 'Cocoa Touch', 'Xcode',
                                'Objective-C', 'SQLite', 'Plist', 'StoreKit', "UI-Kit", 'AV Foundation',
                                'Auto-Layout']
            rec_course = course_recommender(ios_course, 4)
            break
        elif skill_lower in uiux_keyword:
            reco_field = 'UI-UX Development'
            recommended_skills = ['UI', 'User Experience', 'Adobe XD', 'Figma', 'Zeplin', 'Balsamiq',
                                'Prototyping', 'Wireframes', 'Storyframes', 'Adobe Photoshop', 'Editing',
                                'Illustrator', 'After Effects', 'Premier Pro', 'Indesign', 'Wireframe',
                                'Solid', 'Grasp', 'User Research']
            rec_course = course_recommender(uiux_course, 4)
            break
    
    # Resume score calculation
    resume_score = 0
    resume_sections = {
    'Professional Summary': 15,  # More modern than "Objective"
    'Work Experience': 30,      # Most important section
    'Skills': 25,               # Key for applicant tracking systems
    'Education': 15,            # Important but less than experience
    'Projects': 10,             # Valuable but optional
    'Achievements': 5           # Nice to have but not critical
}

    feedback = []
    for section, score in resume_sections.items():
       if section in resume_text:
        # Basic quality checks within the existing structure
        quality_checks = {
            'Professional Summary': 'tailored' in resume_text.lower() and 'summary' in resume_text.lower(),
            'Work Experience': any(word in resume_text.lower() for word in ['managed', 'led', 'developed']),
            'Skills': len(resume_data.get('skills', [])) >= 5,
            'Education': 'degree' in resume_text.lower() or 'university' in resume_text.lower(),
            'Projects': 'technologies' in resume_text.lower() or 'outcomes' in resume_text.lower(),
            'Achievements': any(char.isdigit() for char in resume_text)  # Check for quantifiable achievements
        }
        
        if quality_checks.get(section, False):
            resume_score += score
            feedback.append(f"[+] Excellent! Your '{section}' section looks strong")
        else:
            # Give partial credit if section exists but lacks quality
            resume_score += score * 0.5  
            feedback.append(f"[~] Good start with '{section}', but could be improved")
    else:
        feedback.append(f"[-] Consider adding a '{section}' section (potential +{score} points)")

# Adjust score based on resume length (without new functions)
    if no_of_pages == 1:
       resume_score *= 0.9  # Slightly penalize very short resumes
    elif no_of_pages >= 3:
       resume_score *= 0.8  # Penalize overly long resumes

# Cap the score at 100
    resume_score = min(resume_score, 100)

# Get random videos
    resume_vid = random.choice(resume_videos)
    interview_vid = random.choice(interview_videos)

    return {
    'basic_info': {
        'name': resume_data.get('name'),
        'email': resume_data.get('email'),
        'contact': resume_data.get('mobile_number'),
        'pages': no_of_pages,
        'level': cand_level
    },
    'skills': {
        'existing': resume_data.get('skills', []),
        'recommended': recommended_skills,
        'field': reco_field
    },
    'courses': rec_course,
    'score': round(resume_score),
    'feedback': feedback,
    'videos': {
        'resume': resume_vid,
        'interview': interview_vid
    }
}
@csrf_exempt
@api_view(['POST'])
def upload_resume(request):
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)
    
    uploaded_file = request.FILES['file']
    if not uploaded_file.name.endswith('.pdf'):
        return JsonResponse({'error': 'Only PDF files are allowed'}, status=400)
    
    # Save the file temporarily
    TEMP_DIR = os.path.join(os.getcwd(), 'temp_uploads')
    os.makedirs(TEMP_DIR, exist_ok=True)
    temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
    
    try:
        with open(temp_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        analysis_result = analyze_resume(temp_path)
        if analysis_result:
            return JsonResponse(analysis_result)
        else:
            return JsonResponse({'error': 'Failed to analyze resume'}, status=500)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)





import os
import docx2txt
import PyPDF2
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

@csrf_exempt
def match_resumes(request):
    if request.method == 'POST':
        try:
            # For API requests (React frontend)
            if request.content_type == 'application/json':
                data = json.loads(request.body)
                job_description = data.get('job_description', '')
                resume_files = request.FILES.getlist('resumes')
            else:
                # For form submissions (if you want to keep template rendering)
                job_description = request.POST.get('job_description', '')
                resume_files = request.FILES.getlist('resumes')

            resumes = []
            for resume_file in resume_files:
                # Save file temporarily
                upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, resume_file.name)
                
                with open(file_path, 'wb+') as destination:
                    for chunk in resume_file.chunks():
                        destination.write(chunk)
                
                # Extract text based on file type
                if file_path.endswith('.pdf'):
                    text = extract_text_from_pdf(file_path)
                elif file_path.endswith('.docx'):
                    text = extract_text_from_docx(file_path)
                elif file_path.endswith('.txt'):
                    text = extract_text_from_txt(file_path)
                else:
                    text = ""
                
                resumes.append(text)
                
                # Clean up - remove the temporary file
                os.remove(file_path)

            if not resumes or not job_description:
                return JsonResponse({'error': 'Please upload resumes and enter a job description.'}, status=400)

            # Vectorize job description and resumes
            vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
            vectors = vectorizer.toarray()

            # Calculate cosine similarities
            job_vector = vectors[0]
            resume_vectors = vectors[1:]
            similarities = cosine_similarity([job_vector], resume_vectors)[0]

            # Get top 5 resumes and their similarity scores
            top_indices = similarities.argsort()[-5:][::-1]
            results = [
                {
                    'filename': resume_files[i].name,
                    'score': round(float(similarities[i]), 2)
                } 
                for i in top_indices
            ]

            return JsonResponse({'results': results})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    # GET request - render template if needed
    return render(request, 'matchresume.html')

# Text extraction functions
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()