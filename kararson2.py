from flask import Flask, request, jsonify
import speech_recognition as sr
import os
import tempfile
import boto3
import json
from botocore.exceptions import ClientError
import numpy as np
import soundfile as sf

app = Flask(__name__)

# AWS Erişim anahtarlarını burada tanımlayın
aws_access_key_id = ''  # Buraya gerçek erişim anahtarınızı yazın
aws_secret_access_key = ''  # Buraya gerçek gizli anahtarınızı yazın

# Amazon Bedrock istemcisini ayarlayın
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='eu-central-1',  # Bölgenizi buraya girin
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Claude 3.5 Sonnet model kimliğini belirtin
model_id = 'anthropic.claude-3-5-sonnet-20240620-v1:0'

# Prompt tanımı
prompt = """
Kullanıcının gerçekleştirmek istediği işlemi anlamam gerekiyor. 
Aşağıda üç işlem tanımlanmıştır:

1. **Dosya Taşıma**: Kullanıcının dosya taşıma işlemi gerçekleştirmek istediğini ifade etmesi durumunda veya 'log' kelimesini kullandığında 'dosya taşıma' ifadesini algılamalıyım. Örneğin: 'Dosya taşıma işlemi yapacağım', 'Log dosyasını taşımak istiyorum' gibi.

2. **Ping Atma**: Kullanıcı bir IP adresine ping atmak istediğini belirtirse 'ping atma' ifadesini algılamalıyım. Örneğin: 'Bir IP adresine ping atmak istiyorum' ya da 'Ping işlemi yap' gibi.

3. **Tarih ve Saat**: Kullanıcı tarihi ve saati yazdırmak istediğini belirtirse 'tarih ve saat' ifadesini algılamalıyım. Örneğin: 'Bugünün tarihini öğrenmek istiyorum' veya 'Tarih ve saat bilgisini göster' gibi.

Kullanıcının hangi işlemi yapmak istediğine dair bir ifade aldığımda, lütfen buna karşılık gelen işlem adını ('dosya taşıma', 'ping atma', 'tarih ve saat') döndür. Eğer işlem tanınmazsa 'tanınmayan işlem' ifadesini döndür. Kullanıcıdan gelen girdi: "{user_input}"

Lütfen kullanıcıdan gelen ifadenin net olduğundan emin olun ve her durumda belirtilen üç işlemden uygun olanı belirt başka hiçbir şey eklemeyin.
"""

def invoke_ai_model(user_input):
    request_body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.7,
        "messages": [
            {"role": "user", "content": prompt.format(user_input=user_input)}
        ]
    })

    try:
        print(f"Invoking model: {model_id}")
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=request_body
        )
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"Error: {error_code} - {error_message}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

def dosya_tasima():
    return "Dosya taşıma işlemi başlıyor..."

def ping_atma():
    return "IP adresine ping atma işlemi başlıyor..."

def tarihi_yazdir():
    return "Bugünün tarihi ve saati..."

@app.route('/yes_no', methods=['POST'])
def recognize_yes_no():
    if 'audio' not in request.files:
        return jsonify({"error": "Audio file is required"}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Geçici bir dosya oluştur
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        audio_path = temp_file.name
        audio_file.save(audio_path)

    output_path = audio_path.replace(".wav", "_converted.wav")

    try:
        # Soundfile ile sesi dönüştür
        data, sample_rate = sf.read(audio_path)

        # Eğer örnekleme hızı 16 kHz değilse, 16 kHz'e yeniden örnekle
        if sample_rate != 16000:
            new_sample_rate = 16000
            data = np.interp(
                np.linspace(0, 1, int(len(data) * new_sample_rate / sample_rate)),
                np.linspace(0, 1, len(data)),
                data
            )
            sample_rate = new_sample_rate

        # Dönüştürülmüş dosyayı yaz
        sf.write(output_path, data, sample_rate)
        
        # SpeechRecognition ile sesi metne çevir
        recognizer = sr.Recognizer()
        with sr.AudioFile(output_path) as source:
            audio_data = recognizer.record(source)  # Tüm sesi al
            text = recognizer.recognize_google(audio_data, language='tr-TR')  # Türkçe dil desteği

    except sr.UnknownValueError:
        return jsonify({"error": "Sesi anlayamadım, lütfen tekrar deneyin."}), 400
    except sr.RequestError:
        return jsonify({"error": "Tanıma servisine bağlanırken bir hata oluştu."}), 500
    except Exception as e:
        return jsonify({"error": "Audio conversion failed: " + str(e)}), 500
    finally:
        # Geçici dosyayı sil
        os.remove(audio_path)
        if os.path.exists(output_path):
            os.remove(output_path)

    # Evet veya hayır kontrolü
    text = text.strip().lower()
    
    if 'evet' in text:
        return jsonify({"response": "Evet olarak algılandı."})
    elif 'hayır' in text:
        return jsonify({"response": "Hayır olarak algılandı."})
    else:
        return jsonify({"error": "Yanıt anlaşılmadı. Lütfen tekrar söyler misiniz?"}), 400


@app.route('/convert', methods=['POST'])
def convert_audio_to_text():
    if 'audio' not in request.files:
        return jsonify({"error": "Audio file is required"}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Geçici bir dosya oluştur
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        audio_path = temp_file.name
        audio_file.save(audio_path)

    output_path = audio_path.replace(".wav", "_converted.wav")

    try:
        # Soundfile ile sesi dönüştür
        data, sample_rate = sf.read(audio_path)

        # Eğer örnekleme hızı 16 kHz değilse, 16 kHz'e yeniden örnekle
        if sample_rate != 16000:
            new_sample_rate = 16000
            data = np.interp(
                np.linspace(0, 1, int(len(data) * new_sample_rate / sample_rate)),
                np.linspace(0, 1, len(data)),
                data
            )
            sample_rate = new_sample_rate

        # Dönüştürülmüş dosyayı yaz
        sf.write(output_path, data, sample_rate)
        
        # SpeechRecognition ile sesi metne çevir
        recognizer = sr.Recognizer()
        text = ""
        with sr.AudioFile(output_path) as source:
            audio_data = recognizer.record(source)  # Tüm sesi al
            text = recognizer.recognize_google(audio_data, language='tr-TR')  # Türkçe dil desteği

    except sr.UnknownValueError:
        return jsonify({"error": "Sesi anlayamadım, lütfen tekrar deneyin."}), 400
    except sr.RequestError:
        return jsonify({"error": "Tanıma servisine bağlanırken bir hata oluştu."}), 500
    except Exception as e:
        return jsonify({"error": "Audio conversion failed: " + str(e)}), 500
    finally:
        # Geçici dosyayı sil
        os.remove(audio_path)
        if os.path.exists(output_path):
            os.remove(output_path)

    # AI modelini çağır
    action = invoke_ai_model(text)

    if action:
        action = action.strip().lower()  # Yanıtı küçük harflerle düzenle
        response_message = ""

        # İlgili işlemi çalıştırıyoruz
        if 'dosya taşıma' in action:
            response_message = dosya_tasima()
        elif 'ping atma' in action:
            response_message = ping_atma()
        elif 'tarih ve saat' in action:
            response_message = tarihi_yazdir()
        else:
            response_message = "Üzgünüm, yaptığınız işlem tanınmadı. Lütfen tekrar deneyin."
        
        return jsonify({"text": text, "action": action, "response": response_message})
    else:
        return jsonify({"error": "Modelden geçerli bir yanıt alınamadı."}), 500

if __name__ == '__main__':
    app.run(debug=True)
