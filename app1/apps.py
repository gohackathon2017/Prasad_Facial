from django.apps import AppConfig
from .services import load_models

class App1Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app1'

    def ready(self):
        # Load saved models when the application starts
        load_models()
