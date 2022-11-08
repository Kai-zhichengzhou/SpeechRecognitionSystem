
from django.http import response
from django.test import TestCase, Client
from account.models import User
from django.contrib.auth import SESSION_KEY

class UserModelTest(TestCase):
    @classmethod
    def setUpTestData(cls) -> None:
        #set up the objects that can not be modified by all test case
        User.objects.create(username = "jacky0303", first_name = "Jack", last_name = "Brown", email = "jacky0303@163.com", password = 'feishigogo123')
        
    def test_first_name_label(self):
        user = User.objects.get(id = 1)
        field_label = user._meta.get_field('first_name').verbose_name
        self.assertEqual(field_label, 'first name')

    def test_last_name_label(self):
        user = User.objects.get(id = 1)
        field_label = user._meta.get_field('last_name').verbose_name
        self.assertEqual(field_label, 'last name')

    def test_first_name_max_length(self):
        user = User.objects.get(id = 1)
        max_length = user._meta.get_field('first_name').max_length
        self.assertEqual(max_length, 150)

    def test_last_name_max_length(self):
        user = User.objects.get(id = 1)
        max_length = user._meta.get_field('last_name').max_length
        self.assertEqual(max_length, 150)

    def test_representation_of_object(self):
        user = User.objects.get(id = 1)
        representation_object = f'{user.first_name}, {user.last_name}'
        self.assertEqual(user.display_user(),representation_object)

    def test_username_max_length(self):
        user = User.objects.get(id = 1)
        max_length = user._meta.get_field('username').max_length
        self.assertEqual(max_length, 150)

    def test_username_label(self):
        user = User.objects.get(id = 1)
        field_label = user._meta.get_field('username').verbose_name
        self.assertEqual(field_label, 'username')

    def test_email_max_length(self):
        user = User.objects.get(id = 1)
        max_length = user._meta.get_field('email').max_length
        self.assertEqual(max_length, 254)


    def test_email_label(self):
        user = User.objects.get(id = 1)
        field_label = user._meta.get_field('email').verbose_name
        self.assertEqual(field_label, 'email')

    def test_membership_max_length(self):
        user = User.objects.get(id = 1)
        max_length = user._meta.get_field('membership').max_length
        self.assertEqual(max_length, 1)

    def test_membership_label(self):
        user = User.objects.get(id = 1)
        field_lalel = user._meta.get_field('membership').verbose_name
        self.assertEqual(field_lalel, 'membership')

    def test_auto_generate_date_joined(self):
        user = User.objects.get(id = 1)
        date_joined = user.display_date_joined()
        self.assertIsNotNone(date_joined)

    # def test_login(self):
    # #send login data
    #     c = Client()
    #     response = c.post('login', {'username':'jacky0303', 'password':'feishigogo123'})
    #     self.assertEqual(response.status_code, 200)













    

    




    

    
