from django.core.management.base import BaseCommand
from movie.models import Movie
import os
import numpy as np

import openai


from dotenv import load_dotenv, find_dotenv

class Command(BaseCommand):
    help = 'Modify path of images'

    def handle(self, *args, **kwargs):

        #Se lee del archivo .env la api key de openai
        _ = load_dotenv('../openAI.env')
        openai.api_key  = os.environ['openAI_api_key']
        
        items = Movie.objects.all()

        req = "un joven afroamericano que crece en un barrio conflictivo de Miami"
        emb_req = openai.embeddings.create(input=[req],model='text-embedding-ada-002').data[0].embedding
        print("Pelicula a buscar: ",req)
        sim = []
        
        for i in range(len(items)):
            emb = items[i].emb
            emb = list(np.frombuffer(emb))
            sim.append(np.dot(emb_req, emb)/(np.linalg.norm(emb_req)*np.linalg.norm(emb)))
        sim = np.array(sim)
        idx = np.argmax(sim)
        idx = int(idx)
        print(items[idx].title)