import datetime
from django.test import TestCase
from account.forms import RegisterForm,DocumentForm
class RegistrationFormTest(TestCase):
    def test_register_form_username(self):
        form = RegisterForm()

        self.assertTrue(form.fields['username'].label == 'Username')