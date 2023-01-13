import os, io
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
import pandas as pd
import urllib.request
import pandas as pd

import numpy as np
import string
from datetime import datetime

import pypyodbc

import re

from keras.applications.vgg16 import preprocess_input

import pdf2image
import cv2

import requests
import PyPDF2
import requests_ntlm

from tqdm.notebook import tqdm_notebook

from keras.models import load_model

from credentials import *

import warnings
warnings.filterwarnings('ignore')


import keyring
keyring.set_password(username_,user_path,password_)

r = requests.get(url_, auth=requests_ntlm.HttpNtlmAuth(username_, password_))
print(r.status_code)
r.raise_for_status()
# #r.json()

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = json_file

client = vision_v1.ImageAnnotatorClient()

best_model = load_model('best_model_final.pt/')

def get_prediction(image):
    image = np.expand_dims(image, axis=0)
    prediction = best_model.predict(image)
    predicted_class = np.argmax(prediction)
    return class_names_processed[predicted_class]

def sample_batch_annotate_files_url_auth(file_path_url):
    import requests 
    """Perform batch file annotation."""
    rxcountpages = re.compile(r"/Type\s*/Page([^s]|$)", re.MULTILINE|re.DOTALL)
    client = vision_v1.ImageAnnotatorClient()

    response = requests.get(file_path_url,auth=requests_ntlm.HttpNtlmAuth(username_, keyring.get_password(user_, username_)))
    pdf_file = io.BytesIO(response.content) # response being a requests Response object
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    number = pdf_reader.numPages
    
    pil_images = pdf2image.convert_from_bytes(response.content, dpi=200, 
                 output_folder=None, first_page=None, last_page=None,
                 thread_count=1, userpw=None,use_cropbox=False, strict=False,
                 poppler_path=r"poppler-0.68.0\bin",)
    
    list_predicted_classes = []

    for page in pil_images:

        image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        image_resized = preprocess_input(cv2.resize(image, dsize=(224,224)))
        
        predicted_class = get_prediction(image_resized)
        
        list_predicted_classes.append(predicted_class)

    
    # Supported mime_type: application/pdf, image/tiff, image/gif
    mime_type = "application/pdf"
    #with urllib.request.urlopen(file_path_url) as r:
    #    content = r.read()
    content = response.content

    input_config = {"mime_type": mime_type, "content": content}
    features = [{"type_": vision_v1.Feature.Type.DOCUMENT_TEXT_DETECTION}]

    while number > 0:
        i = 0
        batch = 5
        counter = 0
        page = 1
        j = 1
        lista = []
        for k in range(round(abs(number/5 + 0.5))):
            if number > 5:
                client = vision_v1.ImageAnnotatorClient()
                mime_type = "application/pdf"
#                 features = [{"type_": vision_v1.Feature.Type.DOCUMENT_TEXT_DETECTION}]
                pages=[]
                for page in range(j,batch+1):
                    pages.append(page)
                j+=5
                batch+=5
                requests = [{"input_config": input_config, "features": features, "pages": pages}]
                response = client.batch_annotate_files(requests=requests)
                for image_response in response.responses[0].responses:
                    counter += 1
                    text_lista = image_response.full_text_annotation.text
                    lista.append(text_lista)
                number = number - 5
            else:
                client = vision_v1.ImageAnnotatorClient()
                mime_type = "application/pdf"
                features = [{"type_": vision_v1.Feature.Type.DOCUMENT_TEXT_DETECTION}]
                pages=[]
                for page in range(j,j+number):
                    pages.append(page)
                requests = [{"input_config": input_config, "features": features, "pages": pages}]
                response = client.batch_annotate_files(requests=requests)
                for image_response in response.responses[0].responses:
                    counter += 1
                    text_lista = image_response.full_text_annotation.text
                    lista.append(text_lista)
                number = number - 5
                return list_predicted_classes, lista
            
class_names_processed = ['class_1_Swiss_IDs',
                         'class_2_Other_IDs',
                         'class_3_Driving_SwissPass',
                         'class_4_Text']

regex = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')

def isValid(email):
    if re.fullmatch(regex, email):
        return True
    else:
        return False
    
def Birthday(joined_text):
    dates = re.findall(r'\d{2}.\d{2}.\d{4}', joined_text)

    date_list = []
    for date in dates:
        try:
            if pd.to_datetime(date).year > 1930:
                date_list.append(pd.to_datetime(date, format = '%d.%m.%Y'))
        except:
            continue
            
    return date_list

def check_mobile_number(number):
    
    check_for_numbers = [char.isnumeric() for char in number]
    
    if False not in check_for_numbers and len(check_for_numbers) == 10:
        return True
    else:
        return False
    
cnxn = pypyodbc.connect(r'Driver=SQL Server Native Client 11.0;Server='+server+';Database='+database+';Trusted_Connection=yes;')

data_frq = pd.read_sql_query('''
       SELECT
       scanning.[INVENTTRANSID]
      ,scanning.[CONTRACTCLOSINGDATE]
      ,scanning.[CONTRACTDOCUMENTURL]
     FROM [stage].[table] scanning
     LEFT JOIN [database].[ocr].Provider_name ocr
                ON ocr.InventtransID = scanning.INVENTTRANSID
     WHERE LEN (CONTRACTDOCUMENTURL ) > 1
  and ocr.[INVENTTRANSID] IS NULL
  and PROVIDERNAME ='Provider_name'
  and CONTRACTCLOSINGDATE >= '2022-10-01' ''',cnxn)

my_list = []

for i in range(len(data_frq['contractdocumenturl'])):
    my_list.append((data_frq['contractdocumenturl'][i], data_frq['inventtransid'][i]))
    
    
name_list, surname_list, anrede_list_all, geburtsdatum_list, nationality_list, telefon_list, plz_list, email_list, lang_list, pdf_list, ausweistyp_list, ausweistyp_nummer_list, unsuccessfull, is_valid, InventtransID_list, Class_1, Class_2, Class_3, successfull = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
anrede_list = ['Frau', 'Herr', 'Signora', 'Signor', 'Madame', 'Monsieur']

for i in tqdm_notebook(range(len(my_list))):
    pdf = my_list[i]
    
    try:
        
        predicted_class, text = sample_batch_annotate_files_url_auth(pdf[0])

        text_splitted = ''.join(text).split('\n')
        joined_text = ''.join(text)

        for Class in predicted_class:
            if 'class_1_Swiss_IDs' in predicted_class:
                id_document = 'Valid Document'
                if 'class_2_Other_IDs' not in predicted_class and 'class_3_Driving_SwissPass' not in predicted_class:
                    class_1 = True
                    class_2 = False
                    class_3 = False
                elif 'class_2_Other_IDs' in predicted_class and 'class_3_Driving_SwissPass' not in predicted_class:
                    class_1 = True
                    class_2 = True
                    class_3 = False
                elif 'class_2_Other_IDs' not in predicted_class and 'class_3_Driving_SwissPass' in predicted_class:
                    class_1 = True
                    class_2 = False
                    class_3 = True
                else:
                    class_1 = True
                    class_2 = True
                    class_3 = True
            elif 'class_2_Other_IDs' in predicted_class and 'class_3_Driving_SwissPass' in predicted_class:
                id_document = 'Valid Document'
                class_1 = False
                class_2 = True
                class_3 = True
            else:
                id_document = 'Must Have Second ID'
                if 'class_2_Other_IDs' in predicted_class:
                    class_1 = False
                    class_2 = True
                    class_3 = False
                elif 'class_3_Driving_SwissPass' in predicted_class:
                    class_1 = False
                    class_2 = False
                    class_3 = True
                else:
                    class_1 = False
                    class_2 = False
                    class_3 = False

        if 'Telefono' in joined_text or 'Nazionalità:' in joined_text or 'Tipo di documento di identità:' in joined_text:
            land = 'Ita'
            ##### Email #####
            e_mail = 'Unknown'
            for word in text_splitted:
                if '@' in list(word):
                    e_mail = word
                    break

            if isValid(e_mail) == False and e_mail != 'Unknown' and 'E-Mail:' in text_splitted:
                e_mail = text_splitted[text_splitted.index("E-Mail:") + 1]

            if '@' not in e_mail:
                e_mail = 'Unknown'

            ##### Vorname #####
            try:
                name = re.search('(?<=Vorname)(.*)', joined_text.translate(str.maketrans('','',string.punctuation))).group().strip()
            except:
                name = re.search('(?<=Nome)(.*)', joined_text.translate(str.maketrans('','',string.punctuation))).group().strip()

            #####  Nachname #####
            try:
                surname = re.search('(?<=Nachname)(.*)', joined_text.translate(str.maketrans('','',string.punctuation))).group().strip()
            except:
                surname = re.search('(?<=Cognome)(.*)', joined_text.translate(str.maketrans('','',string.punctuation))).group().strip()

            ##### Anrede #####
            anrede = 'Unknown'

            for word in text_splitted:
                for anrede_ in anrede_list:
                    if word == anrede_:
                        anrede = word

            ##### Birthday #####
            date_list = Birthday(joined_text)

            date = str(min(date_list)).split(' ')[0]
            geburtsdatum = pd.to_datetime(date).strftime('%d.%m.%Y')

            if min(date_list).year >= 2010:
                geburtsdatum = 'Unknown'

            ###### Telefonnumer ######
            if 'Telefono:' in text_splitted:
                mobile = text_splitted[text_splitted.index('Telefono:') + 1]

            for i in range(50):
                if not check_mobile_number(mobile):
                    mobile = text_splitted[text_splitted.index("Telefono:") + 50 - i]
                else:
                    break

            if not check_mobile_number(mobile):
                mobile = 'Unknown'

            ##### PLZ / ORT #####
            plz = re.findall(r'\d{4} ', joined_text)[0] + re.findall(r'\d{4} (.*)', joined_text)[0]

            if plz.replace(' ', '').isnumeric():
                plz = re.search(r'\d{4} [a-zA-Z]+', joined_text).group()

            if 'GB' in plz:
                plz = 'Unknown'

            ##### Nationality #####
            nationality = 'Unknown'
            for i in range(len(text_splitted)):
                if 'Nazionalità:' in text_splitted[i]:
                    nationality = text_splitted[i].split(':')[1].strip()

            ##### Ausweistype #####
            ausweistyp = 'Unknown'
            for i in range(len(text_splitted)):
                if 'Tipo di documento di identità:' in text_splitted[i]:
                    ausweistyp = text_splitted[i].split(':')[1].strip()

            ##### Ausweistyp nummer #####
            ausweistyp_nummer = 'Unknown'
            for i in range(len(text_splitted)):
                if "N. documento d'identità:" in text_splitted[i]:
                    ausweistyp_nummer = text_splitted[i].split(':')[1].strip()

            pdf_list.append(pdf)
            InventtransID_list.append(pdf[1])
            name_list.append(name)
            surname_list.append(surname)
            anrede_list_all.append(anrede)
            geburtsdatum_list.append(geburtsdatum)
            nationality_list.append(nationality)
            telefon_list.append(mobile)
            plz_list.append(plz)
            lang_list.append(land)
            email_list.append(e_mail)
            ausweistyp_nummer_list.append(ausweistyp_nummer)
            ausweistyp_list.append(ausweistyp)
            is_valid.append(id_document)
            Class_1.append(class_1)
            Class_2.append(class_2)
            Class_3.append(class_3)
            successfull.append(1)
        elif 'Téléphone:' in joined_text or 'Nationalité:' in joined_text or "Type de document d'identité:" in joined_text:
            land = 'Fra'

            ##### Email #####
            e_mail = 'Unknown'
            for word in text_splitted:
                if '@' in list(word):
                    e_mail = word
                    break

            if isValid(e_mail) == False and e_mail != 'Unknown' and 'E-Mail:' in text_splitted:
                e_mail = text_splitted[text_splitted.index("E-Mail:") + 1]

            if '@' not in e_mail:
                e_mail = 'Unknown'

            ##### Vorname #####
            try:
                name = re.search('(?<=Vorname)(.*)', joined_text.translate(str.maketrans('','',string.punctuation))).group().strip()
            except:
                name = re.search('(?<=Nom de famille)(.*)', joined_text.translate(str.maketrans('','',string.punctuation))).group().strip()

            #####  Nachname #####
            try:
                surname = re.search('(?<=Nachname)(.*)', joined_text.translate(str.maketrans('','',string.punctuation))).group().strip()
            except:
                surname = re.search('(?<=Prénom)(.*)', joined_text.translate(str.maketrans('','',string.punctuation))).group().strip()

            ##### Anrede #####
            anrede = 'Unknown'

            for word in text_splitted:
                for anrede_ in anrede_list:
                    if word == anrede_:
                        anrede = word

            ##### Birthday #####
            date_list = Birthday(joined_text)

            date = str(min(date_list)).split(' ')[0]
            geburtsdatum = pd.to_datetime(date).strftime('%d.%m.%Y')

            if min(date_list).year >= 2010:
                geburtsdatum = 'Unknown'

            ###### Telefonnumer ######
            if 'Téléphone:' in text_splitted:
                mobile = text_splitted[text_splitted.index('Téléphone:') + 1]

            for i in range(50):
                if not check_mobile_number(mobile):
                    mobile = text_splitted[text_splitted.index("Téléphone:") + 50 - i]
                else:
                    break

            if not check_mobile_number(mobile):
                mobile = 'Unknown'

            ##### PLZ / ORT #####
            plz = re.findall(r'\d{4} ', joined_text)[0] + re.findall(r'\d{4} (.*)', joined_text)[0]

            if plz.replace(' ', '').isnumeric():
                plz = re.search(r'\d{4} [a-zA-Z]+', joined_text).group()

            if 'GB' in plz:
                plz = 'Unknown'

            ##### Nationality #####
            nationality = 'Unknown'
            for i in range(len(text_splitted)):
                if 'Nationalité:' in text_splitted[i]:
                    nationality = text_splitted[i].split(':')[1].strip()

            ##### Ausweistype #####
            ausweistyp = 'Unknown'
            for i in range(len(text_splitted)):
                if "Type de document d'identité:" in text_splitted[i]:
                    ausweistyp = text_splitted[i].split(':')[1].strip()

            ##### Ausweistyp nummer #####
            ausweistyp_nummer = 'Unknown'
            for i in range(len(text_splitted)):
                if "No du document d'identité:" in text_splitted[i]:
                    ausweistyp_nummer = text_splitted[i].split(':')[1].strip()

            pdf_list.append(pdf)
            InventtransID_list.append(pdf[1])
            name_list.append(name)
            surname_list.append(surname)
            anrede_list_all.append(anrede)
            geburtsdatum_list.append(geburtsdatum)
            nationality_list.append(nationality)
            telefon_list.append(mobile)
            plz_list.append(plz)
            lang_list.append(land)
            email_list.append(e_mail)
            ausweistyp_nummer_list.append(ausweistyp_nummer)
            ausweistyp_list.append(ausweistyp)
            is_valid.append(id_document)
            Class_1.append(class_1)
            Class_2.append(class_2)
            Class_3.append(class_3)
            successfull.append(1)
        elif 'Identification type' in joined_text or 'Phone number:' in joined_text or 'Identification no.:' in joined_text:
            land = 'Eng'

            ##### Email #####
            e_mail = 'Unknown'
            for word in text_splitted:
                if '@' in list(word):
                    e_mail = word
                    break

            if isValid(e_mail) == False and e_mail != 'Unknown' and 'E-Mail:' in text_splitted:
                e_mail = text_splitted[text_splitted.index("E-Mail:") + 1]

            if '@' not in e_mail:
                e_mail = 'Unknown'

            ##### Vorname #####
            try:
                name = re.search('(?<=Vorname)(.*)', joined_text.translate(str.maketrans('','',string.punctuation))).group().strip()
            except:
                name = re.search('(?<=First Name)(.*)', joined_text.translate(str.maketrans('','',string.punctuation))).group().strip()

            #####  Nachname #####
            try:
                surname = re.search('(?<=Nachname)(.*)', joined_text.translate(str.maketrans('','',string.punctuation))).group().strip()
            except:
                surname = re.search('(?<=Last Name)(.*)', joined_text.translate(str.maketrans('','',string.punctuation))).group().strip()

            if e_mail != []:
                land = 'Ger'

            ##### Anrede #####
            anrede = 'Unknown'

            for word in text_splitted:
                for anrede_ in anrede_list:
                    if word == anrede_:
                        anrede = word

            ##### Birthday #####
            date_list = Birthday(joined_text)

            date = str(min(date_list)).split(' ')[0]
            geburtsdatum = pd.to_datetime(date).strftime('%d.%m.%Y')

            if min(date_list).year >= 2010:
                geburtsdatum = 'Unknown'

            ###### Telefonnumer ######
            if 'Phone number:' in text_splitted:
                mobile = text_splitted[text_splitted.index('Phone number:') + 1]

            for i in range(50):
                if not check_mobile_number(mobile):
                    mobile = text_splitted[text_splitted.index("Phone number:") + 50 - i]
                else:
                    break

            if not check_mobile_number(mobile):
                mobile = 'Unknown'

            ##### PLZ / ORT #####
            plz = re.findall(r'\d{4} ', joined_text)[0] + re.findall(r'\d{4} (.*)', joined_text)[0]

            if plz.replace(' ', '').isnumeric():
                plz = re.search(r'\d{4} [a-zA-Z]+', joined_text).group()

            if 'GB' in plz:
                plz = 'Unknown'

            #### Nationality ####
            nationality = 'Unknown'
            for i in range(len(text_splitted)):
                if 'Nationality:' in text_splitted[i]:
                    nationality = text_splitted[i].split(':')[1].strip()

            ##### Ausweistype #####
            ausweistyp = 'Unknown'
            for i in range(len(text_splitted)):
                if 'Identification type:' in text_splitted[i]:
                    ausweistyp = text_splitted[i].split(':')[1].strip()

            ##### Ausweistype nummer #####
            ausweistyp_nummer = 'Unknown'
            for i in range(len(text_splitted)):
                if 'Identification no.:' in text_splitted[i]:
                    ausweistyp_nummer = text_splitted[i].split(':')[1].strip()

            pdf_list.append(pdf)
            InventtransID_list.append(pdf[1])
            name_list.append(name)
            surname_list.append(surname)
            anrede_list_all.append(anrede)
            geburtsdatum_list.append(geburtsdatum)
            nationality_list.append(nationality)
            telefon_list.append(mobile)
            plz_list.append(plz)
            lang_list.append(land)
            email_list.append(e_mail)
            ausweistyp_nummer_list.append(ausweistyp_nummer)
            ausweistyp_list.append(ausweistyp)
            is_valid.append(id_document) 
            Class_1.append(class_1)
            Class_2.append(class_2)
            Class_3.append(class_3)
            successfull.append(1)
        elif 'Vorname*:' in text_splitted or 'Vorname' in text_splitted or 'Vorname* :' in text_splitted:
            land = 'Ger'

            ##### Email #####
            e_mail = 'Unknown'
            for word in text_splitted:
                if '@' in list(word):
                    e_mail = word
                    break

            if isValid(e_mail) == False and e_mail != 'Unknown' and 'E-Mail:' in text_splitted:
                e_mail = text_splitted[text_splitted.index("E-Mail:") + 1]

            if '@' not in e_mail:
                e_mail = 'Unknown'

            ##### Vorname #####
            name = re.search('(?<=Vorname)(.*)', joined_text.translate(str.maketrans('','',string.punctuation))).group().strip()

            #####  Nachname #####
            surname = re.search('(?<=Nachname)(.*)', joined_text.translate(str.maketrans('','',string.punctuation))).group().strip()

            if e_mail != []:
                land = 'Ger'

            ##### Anrede #####
            anrede = 'Unknown'

            for word in text_splitted:
                for anrede_ in anrede_list:
                    if word == anrede_:
                        anrede = word

            ##### Birthday #####
            date_list = Birthday(joined_text)

            date = str(min(date_list)).split(' ')[0]
            geburtsdatum = pd.to_datetime(date).strftime('%d.%m.%Y')

            if min(date_list).year >= 2010:
                geburtsdatum = 'Unknown'

            ###### Telefonnumer ######
            if 'Telefon:' in text_splitted:
                mobile = text_splitted[text_splitted.index('Telefon:') + 1]

            for i in range(50):
                if not check_mobile_number(mobile):
                    mobile = text_splitted[text_splitted.index("Telefon:") + 50 - i]
                else:
                    break

            if not check_mobile_number(mobile):
                mobile = 'Unknown'

            ##### PLZ / ORT #####
            plz = re.findall(r'\d{4} ', joined_text)[0] + re.findall(r'\d{4} (.*)', joined_text)[0]

            if plz.replace(' ', '').isnumeric():
                plz = re.search(r'\d{4} [a-zA-Z]+', joined_text).group()

            if 'GB' in plz:
                plz = 'Unknown'

            #### Nationality ####
            nationality = 'Unknown'
            for i in range(len(text_splitted)):
                if 'Nationalität:' in text_splitted[i]:
                    nationality = text_splitted[i].split(':')[1].strip()

            ##### Ausweistype #####
            ausweistyp = 'Unknown'
            for i in range(len(text_splitted)):
                if 'Ausweistyp:' in text_splitted[i]:
                    ausweistyp = text_splitted[i].split(':')[1].strip()

            ##### Ausweistype nummer #####
            ausweistyp_nummer = 'Unknown'
            for i in range(len(text_splitted)):
                if 'Ausweis-Nr.:' in text_splitted[i]:
                    ausweistyp_nummer = text_splitted[i].split(':')[1].strip()

            pdf_list.append(pdf)
            InventtransID_list.append(pdf[1])
            name_list.append(name)
            surname_list.append(surname)
            anrede_list_all.append(anrede)
            geburtsdatum_list.append(geburtsdatum)
            nationality_list.append(nationality)
            telefon_list.append(mobile)
            plz_list.append(plz)
            lang_list.append(land)
            email_list.append(e_mail)
            ausweistyp_nummer_list.append(ausweistyp_nummer)
            ausweistyp_list.append(ausweistyp)
            is_valid.append(id_document) 
            Class_1.append(class_1)
            Class_2.append(class_2)
            Class_3.append(class_3)
            successfull.append(1)
    except:
        pdf_list.append(pdf)
        InventtransID_list.append(pdf[1])
        name_list.append(np.NaN)
        surname_list.append(np.NaN)
        anrede_list_all.append(np.NaN)
        geburtsdatum_list.append(np.NaN)
        nationality_list.append(np.NaN)
        telefon_list.append(np.NaN)
        plz_list.append(np.NaN)
        lang_list.append(np.NaN)
        email_list.append(np.NaN)
        ausweistyp_nummer_list.append(np.NaN)
        ausweistyp_list.append(np.NaN)
        is_valid.append(np.NaN)
        Class_1.append(np.NaN)
        Class_2.append(np.NaN)
        Class_3.append(np.NaN)
        successfull.append(0)
        continue

final_df = pd.DataFrame({
    'URL': pdf_list,
    'InventtransID': InventtransID_list,
    'Vorname': name_list,
    'Name': surname_list,
    'Anrede': anrede_list_all,
    'Geburtsdatum': geburtsdatum_list,
    'Nationality': nationality_list,
    'Telefon': telefon_list,
    'PLZ': plz_list,
    'Sprache': lang_list,
    'email': email_list,
    'Ausweistyp_Nummer': ausweistyp_nummer_list,
    'Ausweistyp': ausweistyp_list,
    'Is_valid': is_valid,
    'Class_1 (Swiss Document)': Class_1,
    'Class_2 (Ausländerausweis)': Class_2,
    'Class_3 (Second Document)': Class_3,
    'Successfull': successfull
})

from datetime import datetime as dt
mask = '%Y%m%d'
date_now = dt.now().strftime(mask)
file_name = file_name
final_df.to_excel(file_name)