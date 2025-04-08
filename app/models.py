from django.db import models
from django.utils.timezone import now
from datetime import timedelta
import uuid
from django.contrib.auth.hashers import make_password

class UserData(models.Model):
    firebase_uid = models.CharField(max_length=128, unique=True, null=True, blank=True)
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=128) 
    token = models.CharField(max_length=255, blank=True, null=True)
    token_expiration = models.DateTimeField(blank=True, null=True)

    is_verified = models.BooleanField(default=False)
    is_google_auth = models.BooleanField(default=False)
    
    def __str__(self):
        return self.name

    def generate_token(self):
        """Generates a new token and sets expiration time."""
        self.token = uuid.uuid4().hex  # Unique token
        self.token_expiration = now() + timedelta(hours=24)  # Token valid for 24 hours
        self.save()

    def is_token_valid(self):
        """Checks if the token is still valid."""
        return self.token and self.token_expiration and self.token_expiration > now()
    
    def set_password(self, raw_password):
        """Ensures proper password hashing"""
        self.password = make_password(raw_password)
        self.save(update_fields=['password'])
    
    def check_password(self, raw_password):
        """Verifies password against hash"""
        from django.contrib.auth.hashers import check_password
        return check_password(raw_password, self.password)
    
    def save(self, *args, **kwargs):
        # Allow empty password for Google auth users
        if self.is_google_auth and not self.password:
            self.password = make_password(None)
        super().save(*args, **kwargs)


class PasswordResetToken(models.Model):
    user = models.ForeignKey(UserData, on_delete=models.CASCADE)
    token = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)
    expired_at = models.DateTimeField()

    def __str__(self):
        return f"Token for {self.user.name}"

class UserProfile(models.Model):
    user = models.OneToOneField(UserData, on_delete=models.CASCADE, related_name='profile')

#personal section
    first_name = models.CharField(max_length=255, blank=True, null=True)
    last_name = models.CharField(max_length=255, blank=True, null=True)
    dob = models.DateField(blank=True, null=True)
    gender = models.CharField(max_length=20, choices=[("male", "Male"), ("female", "Female"), ("other", "Other")], blank=True, null=True)
    country = models.CharField(max_length=255, blank=True, null=True)
    city = models.CharField(max_length=255, blank=True, null=True)
    phone_number = models.CharField(max_length=20, blank=True, null=True)

# professional section
    skills = models.TextField(blank=True, null=True)  # For comma-separated skills
    experience = models.CharField(max_length=100, blank=True, null=True)  # Experience Level: Entry, Mid, Senior
    education = models.CharField(max_length=100, blank=True, null=True)  # Education Level: High School, Bachelor's, etc.
    personal_interests = models.CharField(max_length=255, blank=True, null=True)  # Interests like Technology, AI, etc.
    job_market_demand_score = models.IntegerField(blank=True, null=True)  # Store market demand as an integer

#career goals
    career_transform = models.CharField(max_length=10, null=True, blank=True)  # Yes/No
    previous_job = models.CharField(max_length=255, null=True, blank=True)  # Previous Job Title
    total_experience = models.IntegerField(null=True, blank=True)  # Total Experience in Years
    skill_level = models.CharField(max_length=50, null=True, blank=True)  # Beginner, Intermediate, etc.
    current_skills = models.TextField(null=True, blank=True)  # List of Current Skills
    career_field = models.CharField(max_length=255, null=True, blank=True)  # Target Career Field
    future_goal = models.CharField(max_length=255, null=True, blank=True)  # Future Career Goal
    employment_type = models.CharField(max_length=50, null=True, blank=True)  # Full-time, Part-time, Freelance
    work_preference = models.CharField(max_length=50, null=True, blank=True)  # Remote, On-site, Hybrid
    expected_salary = models.IntegerField(null=True, blank=True)  # Expected Salary in USD
    transition_plan = models.TextField(blank=True, null=True)

    
    # Recommendations stored in JSON format
    recommendations = models.JSONField(default=list)  # To store career recommendations as JSON


    career_name = models.CharField(max_length=255, default="Unknown Career")
    roadmap = models.JSONField(default=dict) 
    projects = models.TextField(blank=True, null=True)  

    def __str__(self):
        return f"{self.user.name}'s Profile"

    

# class CV(models.Model):
#     user = models.OneToOneField(UserData, on_delete=models.CASCADE, related_name="cv")
#     file = models.FileField(upload_to='cvs/')
#     uploaded_at = models.DateTimeField(auto_now_add=True)

    # def __str__(self):
    #     return f"CV for {self.user_profile.user.name}"

class CareerRoadmap(models.Model):
    user = models.OneToOneField(UserData, on_delete=models.CASCADE, related_name="roadmap")
    career_name = models.CharField(max_length=255)
    roadmap = models.JSONField()  # Stores the roadmap as a JSON object
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.email} - {self.career_name}"