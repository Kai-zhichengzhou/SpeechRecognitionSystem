import email
from django.db import models
from django.contrib.auth.models import AbstractUser

# Create your models here.

class User(AbstractUser):
    '''
    A class that build the User Model for storing user's profile information in database

    '''
    email= models.EmailField(unique = True)

    MEMBERSHIP_BRONZE = 'B'
    MEMBERSHIP_PLATINUM = 'P'
    MEMBERSHIP_CHOICE = [
        (MEMBERSHIP_BRONZE, 'Bronze'),
        (MEMBERSHIP_PLATINUM, 'Platinum')
    ]

    phone = models.CharField(max_length=14)
    membership = models.CharField(max_length=1, choices=MEMBERSHIP_CHOICE, default=MEMBERSHIP_BRONZE)
    class Meta(AbstractUser.Meta):
        pass

    def display_user(self):
        return f'{self.first_name}, {self.last_name}'

    def display_date_joined(self):
        return self.date_joined


class TextDocument(models.Model):

    '''
    A class that build the User Model for storing text document's information in database

    '''

    IMPORTANCE_LEVEL_HIGH = "H"
    IMPORTANCE_LEVEL_MEDIUM = "M"
    IMPORTANCE_LEVEL_LOW = "L"
    IMPORTANCE_LEVELS = [
        (IMPORTANCE_LEVEL_HIGH, "High"),
        (IMPORTANCE_LEVEL_MEDIUM, "Medium"),
        (IMPORTANCE_LEVEL_LOW, "Low"),
    ]
    TEXT_TYPE_ARTICLES = "A"
    TEXT_TYPE_ENTERTAINMENT = "E"
    TEXT_TYPE_STUDY = "S"
    TEXT_TYPE_JOURNAL = "J"
    TEXT_TYPE_NOVEL = "N"

    TEXT_CHOICES = [
        (TEXT_TYPE_ARTICLES, "Articles"),
        (TEXT_TYPE_ENTERTAINMENT, "Entertainment"),
        (TEXT_TYPE_STUDY, "Study"),
        (TEXT_TYPE_JOURNAL, "Journal"),
        (TEXT_TYPE_NOVEL, "Novel"),
    ]

    text_content = models.TextField()
    text_level = models.CharField(max_length=1, choices=IMPORTANCE_LEVELS, default=IMPORTANCE_LEVEL_LOW)
    text_tag = models.CharField(max_length=1, choices=TEXT_CHOICES, null = True)
    text_title = models.TextField()
    text_first_create = models.DateTimeField(auto_now_add=True)
    text_last_edit = models.DateTimeField(auto_now = True)
    user = models.ForeignKey(User, models.CASCADE)

