from django.http import response
from django.test import TestCase
from django.utils import timezone
from account.models import User, TextDocument

class TextDocumentsByUserListViewTest(TestCase):
    def setUp(self) -> None:
        
        #create users first
        test_client_1 = User.objects.create(username = "kai_01", first_name = "jacky", last_name = "brown", email = "jacky03@163.com", membership = "B")
        test_client_1.set_password('feishigogo123')

        # test_client_2 = User.objects.create(username = "kai_02", password = "feishigogo123")
        test_client_1.save()
        # test_client_2.save()

        #create text documents
        test_document = TextDocument.objects.create(text_title= "test_article", text_tag = "A", 
                    text_level = "H", text_content = "THis is a test article", user_id =test_client_1.id)
        test_document = TextDocument.objects.create(text_title= "test_note", text_tag = "S", 
                    text_level = "M", text_content = "THis is a test study note", user_id =test_client_1.id)

    def test_login(self):
        login= self.client.login(username = "kai_01", password = "feishigogo123")
        # response = self.client.post('/accounts/login/', {'username':'kai_01', "password":"feishigogo123"})
        response = self.client.get('/accounts/profile/')
        self.assertEqual(str(response.context['profile']), 'kai_01')

    def test_profile_info(self):
        login= self.client.login(username = "kai_01", password = "feishigogo123")
        # response = self.client.post('/accounts/login/', {'username':'kai_01', "password":"feishigogo123"})
        response = self.client.get('/accounts/profile/')
        self.assertEqual(str(response.context['profile']), 'kai_01')
        self.assertEqual(response.context['profile'].first_name, 'jacky')
        self.assertEqual(response.context['profile'].last_name, 'brown')
        self.assertEqual(response.context['profile'].email, 'jacky03@163.com')
        self.assertEqual(response.context['profile'].membership, 'B')



    def test_user_documents_count(self):
        login= self.client.login(username = "kai_01", password = "feishigogo123")
        # response = self.client.post('/accounts/login/', {'username':'kai_01', "password":"feishigogo123"})
        response = self.client.get('/accounts/profile/')
        self.assertEqual(str(response.context['profile']), 'kai_01')
        documents = TextDocument.objects.filter(user_id = response.context['profile'].id)
        print(len(documents))

        self.assertEqual(len(documents), 2)

    def test_verify_all_documents_title(self):
        login= self.client.login(username = "kai_01", password = "feishigogo123")
        # response = self.client.post('/accounts/login/', {'username':'kai_01', "password":"feishigogo123"})
        response = self.client.get('/accounts/profile/')
        self.assertEqual(str(response.context['profile']), 'kai_01')
        documents = TextDocument.objects.filter(user_id = response.context['profile'].id)
        titles = ['test_article', 'test_note']
        for i in range(len(documents)):
            self.assertEqual(documents[i].text_title, titles[i])

    def test_verify_all_documents_content(self):
        login= self.client.login(username = "kai_01", password = "feishigogo123")
        # response = self.client.post('/accounts/login/', {'username':'kai_01', "password":"feishigogo123"})
        response = self.client.get('/accounts/profile/')
        self.assertEqual(str(response.context['profile']), 'kai_01')
        documents = TextDocument.objects.filter(user_id = response.context['profile'].id)
        contents = ['THis is a test article', 'THis is a test study note']
        for i in range(len(documents)):
            self.assertEqual(documents[i].text_content, contents[i])


    def test_verify_documents_tag_and_level(self):
        login= self.client.login(username = "kai_01", password = "feishigogo123")
        # response = self.client.post('/accounts/login/', {'username':'kai_01', "password":"feishigogo123"})
        response = self.client.get('/accounts/profile/')
        self.assertEqual(str(response.context['profile']), 'kai_01')
        documents = TextDocument.objects.filter(user_id = response.context['profile'].id)
        tags = ['A', 'S']
        levels = ['H', 'M']
        for i in range(len(documents)):
            self.assertEqual(documents[i].text_tag, tags[i])
            self.assertEqual(documents[i].text_level ,levels[i])



    def test_create_document(self):
        login= self.client.login(username = "kai_01", password = "feishigogo123")
        # response = self.client.post('/accounts/login/', {'username':'kai_01', "password":"feishigogo123"})
        response = self.client.get('/accounts/profile/')
        self.assertEqual(str(response.context['profile']), 'kai_01')

        documents = TextDocument.objects.filter(user_id = response.context['profile'].id)
        previous_doc_count = len(documents)
        #create new document
        response = self.client.post('/accounts/profile/', {"title": "New Document", "tag":"S", "level":"H", "documents":"This is random pagraph"})
        documents = TextDocument.objects.filter(user_id = response.context['profile'].id)
        
        #test if the post request create a new document in databse
        self.assertEqual(len(documents), previous_doc_count + 1)

    
    def test_speech_recognition_executable(self):
        login= self.client.login(username = "kai_01", password = "feishigogo123")
        # response = self.client.post('/accounts/login/', {'username':'kai_01', "password":"feishigogo123"})
        response = self.client.get('/accounts/profile/')
        self.assertEqual(str(response.context['profile']), 'kai_01')

        #call the view function to call the speech recognition module
        response = self.client.get('/accounts/profile/speech/')

        #test if the response is valid 
        self.assertEqual(response.status_code, 200)
       





