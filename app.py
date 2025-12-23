from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# تحميل النموذج المدرب
try:
    model = joblib.load('bus_delay_model.pkl')
    print("✅ Model loaded successfully!")
except:
    print("❌ Error: Model file not found!")
    model = None

@app.route('/')
def home():
    """الصفحة الرئيسية"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API للتنبؤ"""
    try:
        # استقبال البيانات من الفورم
        data = request.json
        
        # استخراج القيم
        route = data['route']
        weather = data['weather']
        passengers = int(data['passengers'])
        hour = int(data['hour'])
        day = int(data['day'])
        lat = float(data['latitude'])
        lon = float(data['longitude'])
        
        # حساب المتغيرات الإضافية
        is_weekend = 1 if day >= 5 else 0
        is_rush_hour = 1 if (7 <= hour <= 9 or 16 <= hour <= 19) else 0
        is_night = 1 if (hour >= 22 or hour <= 5) else 0
        rush_weekend_interaction = is_rush_hour * is_weekend
        
        # تحديد نوع الوقت
        if 7 <= hour <= 9:
            peak_type = 'morning_peak'
        elif 16 <= hour <= 19:
            peak_type = 'evening_peak'
        else:
            peak_type = 'off_peak'
        
        # إنشاء DataFrame بنفس ترتيب features في التدريب
        input_data = pd.DataFrame([{
            'route_std': route,
            'weather_std': weather,
            'passenger_clean': passengers,
            'lat_clean': lat,
            'lon_clean': lon,
            'hour': hour,
            'day_of_week': day,
            'is_weekend': is_weekend,
            'is_rush_hour': is_rush_hour,
            'is_night': is_night,
            'peak_type': peak_type,
            'rush_weekend_interaction': rush_weekend_interaction
        }])
        
        # التنبؤ
        if model is not None:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            # احتمالية التأخير (class 1)
            delay_prob = probability[1] * 100
            
            return jsonify({
                'success': True,
                'is_late': int(prediction),
                'probability': round(delay_prob, 1),
                'peak_type': peak_type
            })
        else:
            # لو النموذج مش موجود، استخدم قواعد بسيطة
            delay_prob = 30.0
            
            if is_rush_hour:
                delay_prob += 25
            if weather == 'rainy':
                delay_prob += 20
            if passengers > 40:
                delay_prob += 15
            if is_weekend:
                delay_prob -= 10
            if is_night:
                delay_prob -= 15
                
            delay_prob = max(5, min(95, delay_prob))
            
            return jsonify({
                'success': True,
                'is_late': 1 if delay_prob > 50 else 0,
                'probability': round(delay_prob, 1),
                'peak_type': peak_type,
                'note': 'Using fallback prediction (model not loaded)'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    print("🚀 Starting Bus Delay Prediction Server...")
    print("📍 Open: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
