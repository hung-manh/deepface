import os
import re
from datetime import datetime, timedelta
import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import PIL.Image
import numpy as np
from multiprocessing import Process, Queue, Manager
import cv2
from deepface import DeepFace
from deepface.models.FacialRecognition import FacialRecognition


model_name = "Facenet512"
model: FacialRecognition = DeepFace.build_model(model_name=model_name)

class FaceES:
    def __init__(self, folder_avatar: str):
        self.__folder_avatar = folder_avatar
        self.es = Elasticsearch(hosts=['http://localhost:9200'])
        self.es_index = 'labfaces'
        self.facerg = model

    def del_index(self):
        index_exists = self.es.indices.exists(index=self.es_index)
        if index_exists:
            self.es.indices.delete(index=self.es_index)
            print('delete index', self.es_index)

    def create_index(self):
        self.del_index()
        settings = {'number_of_replicas': 0,
                    'refresh_interval': '1m'}
        body = {
            "mappings":{
                "properties": {
                    "fullname": {"type": "keyword"},
                    "face_encoding": {
                        "type": "dense_vector",
                        "dims": 512
                        }
                }
            }
        }
        self.es.indices.create(index=self.es_index,  body=body)
        print('MAPPING OK==', self.es_index)

    def __push2db(self, data: list, refresh: bool = False):
        bulk_data = []
        for i, dr in enumerate(data):
            bulk_data.append(
                {
                    "_index": self.es_index,
                    "_id": i,
                    "_source": {
                        "fullname": dr["name"],
                        "face_encoding": dr["encoding"]
                    }
                }
            )
        if bulk_data:
            bulk(self.es, bulk_data)
        self.es.indices.refresh(index= self.es_index)
        print(self.es.cat.count(index=self.es_index, format="json"))

    def push_avatar_to_es(self):
        known_face_encodings = []
        for r_ in os.listdir(self.__folder_avatar):
            if not r_.startswith('.'):
                r_ = r_.strip()
                print('push_avatar_to_es', r_)

                fileio = os.path.join(self.__folder_avatar, r_)
                
                # Load image file to numpy array 
                img = DeepFace.extract_faces(img_path=fileio, detector_backend='retinaface')[0]["face"]
                img = cv2.resize(img, (160, 160))
                img = np.expand_dims(img, axis=0)
                face_encodings = model.forward(img)

                known_face_encodings.append({'name': r_, 'encoding': face_encodings})
                    
                    
        self.__push2db(data=known_face_encodings, refresh=True)

    def training(self):
        self.del_index()
        self.create_index()
        self.push_avatar_to_es()

    def query(self, vector_encoding: list, delta: float = 0.95) -> str:
        res = []
        if vector_encoding:
            es_result = self.es.search(index=self.es_index, size=1, query={'script_score': {'query': {'match_all': {}},
                                                                                            'script': {'source': "cosineSimilarity(params.query_vector, 'face_encoding')",
                                                                                                       'params': {'query_vector': vector_encoding}}}
                                                                           })
            for dr in es_result['hits']['hits']:
                score = float(dr['_score'])
                if score > delta:
                    res.append(dr['_source']['fullname'][:-4])
        return 'or'.join(res)
        

class HelloFace(FaceES):
    def __init__(self):
        FaceES.__init__(self, folder_avatar='./data/hauiavatar')
        self.b_queue = Queue()
        self._facerg = model

    def process_frame(self,frameimg): 
        now = datetime.now()
        face_locations = self._facerg.face_locations(frameimg)
        all_fullname = []
        
        if face_locations:
            face_encodings = self._facerg.face_encodings(frameimg, face_locations)
            for face_encoding in face_encodings:
                fullname = (self.query(face_encoding.tolist()))
                print('Xin chào %s' % fullname, datetime.now() - now)
                all_fullname.append(fullname)
        return all_fullname


if __name__ == '__main__':
    hf = HelloFace()
    hf.training()

