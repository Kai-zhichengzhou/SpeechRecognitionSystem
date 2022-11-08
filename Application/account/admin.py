from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User
# Register your models here.

# @admin.register(User)
# class UserAdmin(BaseUserAdmin): #inherit the module useradmin
#     add_fieldsets =  (
#         (None, {
#             'classes': ('wide',),
#             'fields':('username','password1', 'password2','email', 'first_name','last_name'),
#         }),
#     )
from django.contrib import admin
from .models import User

admin.site.register(User)