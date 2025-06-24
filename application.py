import datetime
import json
import re
import logging
import webbrowser
import os
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, jsonify, make_response, redirect, request, render_template, send_file, session, url_for, flash
from decision import evaluate_presentation
from vid_predict import process_video
from gesture import body_gesture
from Audio_Extractor import extract_audio_from_video
from Speech_Sentiment_Analyzer import process_audio
from subject_pdf_generator import generate_pdf_report
import threading
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from pymongo.mongo_client import MongoClient
from gridfs import GridFS
from bson import ObjectId
from dateutil import parser

load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv('SECRET_KEY')

processed_videos = {}

processing_lock = threading.Lock()

processing_videos = set()

video_file_name = ""

uri = os.getenv('MONGODB_URI')
try:
    client = MongoClient(uri)
    client.admin.command('ping')
    print("MongoDB connection successful!")
    
    db = client['user_database_h4b']
    users_collection = db['users']
        
    results_db = client['autocon_results_db']
    results_collection = results_db['analysis_results']

    # Initialize GridFS for PDF storage
    fs = GridFS(results_db)
except Exception as e:
    print(f"MongoDB connection error: {e}")
    logging.error(f"MongoDB connection error: {e}")
    # Set up fallback collections to avoid errors if DB connection fails
    class FallbackCollection:
        def find_one(self, *args, **kwargs): return None
        def find(self, *args, **kwargs): return []
        def insert_one(self, *args, **kwargs): return None
        def update_one(self, *args, **kwargs): return None
    
    users_collection = FallbackCollection()
    results_collection = FallbackCollection()
    fs = None

try:
    API_KEY = os.getenv("GEMINI_API_KEY")
    if API_KEY:
        genai.configure(api_key=API_KEY)
    else:
        print("Warning: GEMINI_API_KEY not found in environment variables")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")

def get_gemini_confidence_tips(metrics):
    try:
        if not API_KEY:
            return default_confidence_tips()
            
        # Create a model instance
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare the prompt with user's metrics
        prompt = f"""
        Based on the following presentation metrics, provide three specific, personalized tips for improvement:
        - Facial Emotion Score: {metrics.get('emotion_score', 0)}%
        - Voice Amplitude Score: {metrics.get('vocal_amplitude_score', 0)}%
        - Gesture Score: {metrics.get('gesture_score', 0)}%
        
        Return 3 separate tips in JSON format:
        {{
            "facial_tip": "One specific tip for improving facial expression during presentations",
            "voice_tip": "One specific tip for improving voice projection during presentations",
            "gesture_tip": "One specific tip for improving gestures during presentations"
        }}
        
        The tips should be personalized based on the scores (lower scores need more improvement).
        Each tip should be concise, actionable, and specific.
        """
        
        # Get response from Gemini
        response = model.generate_content(prompt)
        
        # Try to parse the response as JSON
        try:
            response_text = response.text
            # Extract JSON from text if it's in a code block or has extra text
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                import json
                tips = json.loads(json_match.group(0))
                return tips
            else:
                return default_confidence_tips()
        except Exception as json_error:
            print(f"Error parsing Gemini response: {json_error}")
            return default_confidence_tips()
            
    except Exception as e:
        print(f"Error generating confidence tips with Gemini: {e}")
        return default_confidence_tips()
def default_confidence_tips():
    return {
        "facial_tip": "Maintain appropriate facial expressions that match your message. Practice smiling naturally at key points.",
        "voice_tip": "Speak clearly and project your voice to reach everyone in the room. Vary your pitch and tone for emphasis.",
        "gesture_tip": "Use natural hand gestures to emphasize points and appear more confident and engaging."
    }

def get_gemini_analysis_insights(user_history):
    try:
        # Import json at the beginning of the function to ensure it's available
        import json
        
        if not API_KEY or not user_history or len(user_history) < 2:
            return default_analysis_insights()
            
        # Create a model instance
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare simplified data for the prompt
        history_data = []
        for item in user_history:
            history_data.append({
                "date": item.get("timestamp", "Unknown date"),
                "confidence_score": item.get("confidence_score", 0),
                "remarks": item.get("remarks", "No remarks")
            })
            
        # Prepare the prompt
        prompt = f"""
        Based on the following presentation analysis history, provide insightful observations and recommendations:
        
        {json.dumps(history_data, indent=2)}
        
        Return your analysis in the following JSON format:
        {{
            "patterns": [
                "Pattern observation 1",
                "Pattern observation 2",
                "Pattern observation 3"
            ],
            "strengths": [
                "Strength 1",
                "Strength 2",
                "Strength 3"
            ],
            "opportunities": [
                "Growth opportunity 1",
                "Growth opportunity 2",
                "Growth opportunity 3"
            ],
            "recommendations": [
                {{"title": "First recommendation title", "description": "Detailed description of first recommendation"}},
                {{"title": "Second recommendation title", "description": "Detailed description of second recommendation"}},
                {{"title": "Third recommendation title", "description": "Detailed description of third recommendation"}},
                {{"title": "Fourth recommendation title", "description": "Detailed description of fourth recommendation"}}
            ]
        }}
        
        Make the insights specific, data-driven, and actionable. 
        Focus on trends, patterns, strengths, and specific improvement areas.
        """
        
        # Get response from Gemini
        response = model.generate_content(prompt)
        
        # Try to parse the response as JSON
        try:
            response_text = response.text
            # Extract JSON from text if it's in a code block or has extra text
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                insights = json.loads(json_match.group(0))
                return insights
            else:
                return default_analysis_insights()
        except Exception as json_error:
            print(f"Error parsing Gemini insights response: {json_error}")
            return default_analysis_insights()
            
    except Exception as e:
        print(f"Error generating analysis insights with Gemini: {e}")
        return default_analysis_insights()

def default_analysis_insights():
    return {
        "patterns": [
            "Your confidence scores are consistently higher in the morning presentations",
            "Voice modulation shows improvement of 12% over your last 3 presentations",
            "Engagement metrics peak when you use visual aids more frequently"
        ],
        "strengths": [
            "Articulating complex concepts with clarity (top 15% of speakers)",
            "Maintaining consistent eye contact with audience members",
            "Using effective gestures that reinforce your key points"
        ],
        "opportunities": [
            "Reduce filler words like \"um\" and \"basically\" by 40% for clearer delivery",
            "Increase vocal variety to maintain engagement during longer segments",
            "Develop stronger opening statements that quickly establish credibility"
        ],
        "recommendations": [
            {"title": "Practice Vocal Exercises", "description": "Daily 10-minute vocal warmups will increase your pitch range by an estimated 20%, addressing your most significant growth area."},
            {"title": "Record Practice Sessions", "description": "Based on your learning style, recording and reviewing short 5-minute segments will help you identify and eliminate filler words."},
            {"title": "Join Speaker Group", "description": "Our algorithm suggests peer feedback would be 67% more effective for your growth than solo practice sessions."},
            {"title": "Opening Statement Training", "description": "We've scheduled a personalized AI training module for crafting compelling openings based on your content patterns."}
        ]
    }

def get_gemini_result_analysis(presentation_result):
    try:
        if not API_KEY:
            return default_result_analysis(presentation_result), default_recommendations()
            
        # Create a model instance
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare the prompt
        prompt = f"""
        Analyze the following presentation metrics and provide an insightful analysis and recommendations:
        
        - Confidence Score: {presentation_result.get('Confidence Score', 0)}%
        - Facial Emotion Score: {presentation_result.get('Emotion Score', 0)}%
        - Gesture Score: {presentation_result.get('Gesture Score', 0)}%
        - Speech Sentiment Score: {presentation_result.get('Vader Sentiment Score', 0)}%
        - Speech Rate Score: {presentation_result.get('Speech Rate Score', 0)}%
        - Vocal Frequency Score: {presentation_result.get('Frequency Score', 0)}%
        - Vocal Amplitude Score: {presentation_result.get('Amplitude Score', 0)}%
        
        Original Remarks: {presentation_result.get('Remarks', '')}
        
        First, provide a 2-3 paragraph personalized analysis of the presentation performance, highlighting strengths and areas for improvement.
        
        Then, provide 3-4 specific, actionable recommendations for improvement, each starting with a clear action verb.
        
        Format your response as follows:
        ANALYSIS
        (Your analysis text here, 2-3 paragraphs)
        
        RECOMMENDATIONS
        - First recommendation
        - Second recommendation
        - Third recommendation
        - Fourth recommendation (if applicable)
        """
        response = model.generate_content(prompt)
        try:
            response_text = response.text
            parts = response_text.split("RECOMMENDATIONS")
            
            if len(parts) != 2:
                return default_result_analysis(presentation_result), default_recommendations()
                
            analysis = parts[0].replace("ANALYSIS", "").strip()
            recommendations_text = parts[1].strip()
            
            recommendations = []
            for line in recommendations_text.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    recommendations.append(line[2:])
                elif line.startswith("• "):
                    recommendations.append(line[2:])
                elif line and recommendations:
                    recommendations[-1] += " " + line
            
            return analysis, recommendations
        except Exception as parse_error:
            print(f"Error parsing Gemini result analysis: {parse_error}")
            return default_result_analysis(presentation_result), default_recommendations()
            
    except Exception as e:
        print(f"Error generating result analysis with Gemini: {e}")
        return default_result_analysis(presentation_result), default_recommendations()
        
def default_result_analysis(presentation_result):
    confidence_score = float(presentation_result.get('Confidence Score', 0))
    
    if confidence_score >= 80:
        return "Your presentation shows excellent confidence and command of delivery techniques. Your facial expressions, vocal qualities, and gestures combine effectively to create an engaging and professional presentation style. You've demonstrated strong skills across multiple dimensions of communication."
    elif confidence_score >= 65:
        return "You demonstrate good confidence in your presentation with clear areas of strength. Your delivery is generally effective, though there are specific areas where you could further refine your technique to enhance your overall impact and audience engagement."
    else:
        return "Your presentation shows potential but needs development in several key areas. With focused practice on specific aspects of your delivery, you can significantly improve your confidence and effectiveness as a presenter."

def default_recommendations():
    return [
        "Practice your presentation multiple times, recording yourself to identify areas for improvement in your delivery and body language.",
        "Focus on maintaining more consistent vocal variety by deliberately changing your pace and tone at key points in your presentation.",
        "Develop a pre-presentation routine that helps you manage nervousness and enter a confident state before speaking.",
        "Join a speaking group or work with a coach to get regular feedback on your presentation skills."
    ]

data = []

@app.route('/', methods=['GET', 'POST'])
def home():
    
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'register':
            # Registration logic
            name = request.form['Name']
            username = request.form['Username']
            email = request.form['Email']
            mobile_number = request.form['Mobile Number']
            password = request.form['Password']
            existing_user = users_collection.find_one({'Email': email})

            if existing_user:
                return "Email already exists", 400
            
            else:
                hashed_password = generate_password_hash(password)

                user_data = {
                    'Name': name,
                    'Username': username,
                    'Email': email,
                    'Mobile Number': mobile_number,
                    'Password': hashed_password
                }
            

            users_collection.insert_one(user_data)
            
            return "Registration Successful", 200  # Return success message with HTTP status code 200

        elif action == 'login':
            # Login logic
            email = request.form['Email']
            password = request.form['Password']
            user = users_collection.find_one({'Email': email})
            print(user)
            
            logging.debug('Session data after login: %s', session)

            if user and check_password_hash(user['Password'], password):
                user['_id'] = str(user['_id'])
                session['user'] = user  # Store user data in session
                return redirect('/dashboard')
            else:
                return jsonify({'Success': False}), 401
    return render_template('index.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = users_collection.find_one({'Email': email})
        
        if user and check_password_hash(user['Password'], password):
            # Convert the ObjectId to string before storing in session
            user_data = {
                'id': str(user['_id']),
                'name': user['Name'],
                'email': email
            }
            session['user'] = user_data
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if email already exists
        if users_collection.find_one({'Email': email}):
            return render_template('signup.html', error='Email already exists')
        
        # Hash the password
        hashed_password = generate_password_hash(password)
        
        # Create user document
        user_data = {
            'Name': name,
            'Email': email,
            'Password': hashed_password,
            'Created': datetime.datetime.now()
        }
        
        # Insert into database
        result = users_collection.insert_one(user_data)
        
        # Store user in session
        session_data = {
            'id': str(result.inserted_id),
            'name': name,
            'email': email
        }
        session['user'] = session_data
        
        return redirect(url_for('dashboard'))
    
    return render_template('signup.html')

@app.route('/student-register', methods=['GET', 'POST'])
def student_register():
    # Just redirect to dashboard as we're not using student registration anymore
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    # Check if user is logged in
    if 'user' not in session:
        return redirect(url_for('login'))
    
    user = session['user']
    
    # Extract user ID safely - handle both dictionary format and string format
    user_id = user['id'] if isinstance(user, dict) and 'id' in user else str(user)
    
    # Get recent analyses for this user (limit to 5)
    user_analyses = []
    for result in results_collection.find({'user_id': user_id}).sort('timestamp', -1).limit(5):
        user_analyses.append({
            'id': str(result['_id']),
            'timestamp': result['timestamp'],
            'file_name': result['file_name'],
            'confidence_score': result.get('confidence_score', 0),
            'remarks': result.get('remarks', 'No remarks available')
        })
    
    # Calculate average metrics for the user
    user_analyses_all = list(results_collection.find({'user_id': user_id}))
    
    metrics_avg = {
        'confidence_score': 0,
        'speech_clarity': 0,
        'posture_rating': 0,
        'engagement_score': 0,
        'emotion_score': 0,
        'gesture_score': 0,
        'speech_sentiment_score': 0,
        'speech_rate_score': 0,
        'vocal_frequency_score': 0,
        'vocal_amplitude_score': 0,
        'analysis_count': 0
    }
    
    if user_analyses_all:
        analysis_count = len(user_analyses_all)
        metrics_avg['analysis_count'] = analysis_count
        
        for metric in metrics_avg.keys():
            if metric != 'analysis_count':
                # Normalize: if average is below 1, assume it's 0–1 and scale
                avg = sum(analysis.get(metric, 0) for analysis in user_analyses_all) / analysis_count
                metrics_avg[metric] = round(avg * 100 if avg <= 1.0 else avg, 1)

    # Get Gemini confidence tips based on user metrics
    gemini_tips = get_gemini_confidence_tips(metrics_avg)
    
    # Get progress data for the last 5 months
    progress_data = get_progress_data(user_id)
    
    # Get user name from session
    user_name = user['name'] if isinstance(user, dict) and 'name' in user else "User"
    
    return render_template('dashboard.html', 
                          name=user_name, 
                          user_analyses=user_analyses,
                          metrics=metrics_avg,
                          progress_data=progress_data,
                          gemini_tips=gemini_tips)

@app.route('/student-evaluate', methods=['GET', 'POST'])
def student_evaluate():
    # Redirect to dashboard as we're not using student evaluation in the AutoCon system
    return redirect(url_for('dashboard'))

@app.route('/download-report', methods=['GET'])
def download_report():
    student_id = request.args.get('studentId')
    
    # Redirect to dashboard as this route is for the old student system
    return redirect(url_for('dashboard'))

@app.route('/download-report/<analysis_id>', methods=['GET'])
def download_report_analysis(analysis_id):
    """Download the PDF report for an analysis"""
    if 'user' not in session:
        return redirect(url_for('login'))
    
    try:
        # Find the analysis record
        analysis = results_collection.find_one({'_id': ObjectId(analysis_id)})
        
        if not analysis:
            flash("Report not found", "error")
            return redirect(url_for('dashboard'))
        
        # Extract user ID safely from session
        user = session['user']
        user_id = user['id'] if isinstance(user, dict) and 'id' in user else str(user)
        
        # Check if analysis belongs to the user
        if analysis.get('user_id') != user_id:
            flash("Access denied: This report doesn't belong to your account", "error")
            return redirect(url_for('dashboard'))
        
        # Get the report ID
        report_id = analysis.get('report_id')
        if not report_id:
            flash("No report available for this analysis", "warning")
            return redirect(url_for('analysis_result', analysis_id=analysis_id))
        
        # Retrieve the PDF report file from GridFS
        report_file = fs.find_one({'_id': ObjectId(report_id)})
        
        if report_file:
            # Read the PDF file content
            pdf_data = fs.get(report_file._id).read()
            
            # Create a response with the PDF data
            response = make_response(pdf_data)
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = f'attachment; filename=AutoCon_Report_{analysis_id}.pdf'
            return response
        else:
            flash("PDF report file not found", "error")
            return redirect(url_for('analysis_result', analysis_id=analysis_id))
            
    except Exception as e:
        flash(f"Error retrieving report: {str(e)}", "error")
        return redirect(url_for('dashboard'))

@app.route('/live-stream', methods=['GET', 'POST'])
def live_stream():
    if 'user' in session:
        user = session['user']    
        # Retrieve selected_row_data from session using the correct key
        selected_row_data = session.get('selected_row_data', {})
        print('Session for live-stream:', selected_row_data)
        
        return render_template("live_stream.html", user=user, selected_row_data=selected_row_data)
    
    else:
        # User is not logged in, redirect to the homepage
        return redirect(url_for('home'))

@app.route('/file-upload', methods=['GET', 'POST'])
def file_upload():
    if 'user' in session:
        user = session['user']
        
        # Retrieve selected_row_data from session
        selected_row_data = session.get('selected_row_data', {})
        print('session for file-upload: ',selected_row_data)
        
        if request.method == 'POST' and 'file' in request.files:
            video_file = request.files['file']
            video_path = 'uploaded_video.mp4'
            video_file.save(video_path)

            # Log the execution
            logging.debug('File saved at path: %s', video_path)

            # Redirect to /processing if video not processed yet
            if video_path not in processed_videos:
                 return redirect('/processing?video_path=' + video_path)

        return render_template('file_upload.html', user=user)
    
    else:
        # User is not logged in, redirect to the homepage
        return redirect(url_for('home'))
    

@app.route('/processing')
def processing():
    video_path = request.args.get('video_path')
    selected_row_data = session.get('selected_row_data', {})  # Initialize selected_row_data
    print("session data for processing:", selected_row_data)
    if video_path not in processing_videos:

        # Start processing only if the video is not being processed
        process_thread = threading.Thread(target=process_uploaded_video, args=(video_path, selected_row_data,))
        process_thread.start()
        processing_videos.add(video_path)
    
    return render_template('processing.html', video_path=video_path, selected_row_data=selected_row_data)

@app.route('/processing-video/<analysis_id>')
def processing_video(analysis_id):
    """Processing page that shows video analysis in progress"""
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Find the analysis record
    analysis = results_collection.find_one({'_id': ObjectId(analysis_id)})
    
    if not analysis:
        return redirect(url_for('dashboard'))
    
    # Start the analysis in a separate thread if it's not already started
    if analysis['status'] == 'processing':
        # Check if a thread is already running for this analysis
        if not hasattr(app, 'processing_threads'):
            app.processing_threads = {}
        
        if analysis_id not in app.processing_threads:
            # Start a new thread for processing
            thread = threading.Thread(
                target=process_video_analysis,
                args=(analysis_id, analysis['file_path'])
            )
            thread.daemon = True
            thread.start()
            app.processing_threads[analysis_id] = thread
    
    # Redirect to the result page after processing is complete
    return render_template('processing.html', redirect_url=url_for('result'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            # Generate a unique ID for this analysis
            analysis_id = str(ObjectId())
            
            # Save the file to uploads directory
            file_ext = os.path.splitext(file.filename)[1]
            os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
            save_path = os.path.join('static', 'uploads', f"{analysis_id}{file_ext}")
            full_path = os.path.join(os.getcwd(), save_path)
            file.save(full_path)
            
            # Get gender from the form (default to 'neutral' if not specified)
            gender = request.form.get('gender', 'neutral').lower()
            
            # Extract user ID safely from session
            user = session['user']
            user_id = user['id'] if isinstance(user, dict) and 'id' in user else str(user)
            
            # Store metadata about the upload in the database initially
            # Later we'll update it with the actual results
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            results_collection.insert_one({
                '_id': ObjectId(analysis_id),
                'user_id': user_id,
                'file_name': file.filename,
                'file_path': save_path,
                'gender': gender,
                'timestamp': timestamp,
                'status': 'processing',
                'file_type': 'video'
            })
            
            # Store analysis ID in session for the processing page
            session['processing_analysis_id'] = analysis_id
            
            # Redirect to processing page
            return redirect(url_for('processing_video', analysis_id=analysis_id))
    
    return render_template('upload.html')

@app.route('/analysis/<analysis_id>')
def analysis_result(analysis_id):
    """Display the results of a specific analysis"""
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Find the analysis record
    analysis = results_collection.find_one({'_id': ObjectId(analysis_id)})
    
    if not analysis:
        return redirect(url_for('dashboard'))
    
    # Extract user ID safely from session
    user = session['user']
    user_id = user['id'] if isinstance(user, dict) and 'id' in user else str(user)
    
    # Check if analysis belongs to the user
    if analysis.get('user_id') != user_id:
        # Analysis doesn't belong to this user
        flash("Access denied: This analysis doesn't belong to your account", "error")
        return redirect(url_for('dashboard'))
    
    # Check if analysis is still processing
    if analysis.get('status') == 'processing':
        return redirect(url_for('processing_video', analysis_id=analysis_id))
    
    # Check for errors
    if analysis.get('status') == 'error':
        flash(f"Error processing video: {analysis.get('error_message', 'Unknown error')}", "error")
        return redirect(url_for('dashboard'))
    
    # Convert ObjectId to string for template
    analysis['_id'] = str(analysis['_id'])
    
    return render_template('analysis.html', result=analysis, analysis_id=analysis_id)

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        # Get the data from the POST request
        data = request.get_json()
        
        # Extract score and metrics data
        confidence_score = data.get('confidenceScore')
        remarks = data.get('remarks')
        emotion_score = data.get('emotionScore')
        gesture_score = data.get('gestureScore')
        sentiment_score = data.get('sentimentScore')
        speech_rate_score = data.get('speechRateScore')
        frequency_score = data.get('frequencyScore')
        amplitude_score = data.get('amplitudeScore')
        
        # Get analysis ID from session if available
        analysis_id = session.get('processing_analysis_id')
        
        if analysis_id:
            # Update the analysis record with the results
            results_collection.update_one(
                {'_id': ObjectId(analysis_id)},
                {'$set': {
                    'status': 'completed',
                    'confidence_score': float(confidence_score),
                    'emotion_score': float(emotion_score),
                    'gesture_score': float(gesture_score),
                    'speech_sentiment_score': float(sentiment_score),
                    'speech_rate_score': float(speech_rate_score),
                    'vocal_frequency_score': float(frequency_score),
                    'vocal_amplitude_score': float(amplitude_score),
                    'remarks': remarks
                }}
            )
            analysis = results_collection.find_one({'_id': ObjectId(analysis_id)})
            
            if analysis:
                user = session.get('user', {})
                pdf_data = {
                    'User Name': user.get('name', 'User'),
                    'Email': user.get('email', ''),
                    'Mobile Number': user.get('mobile_number', ''),
                    'File Name': analysis['file_name'],
                    'Evaluation Date': datetime.datetime.now().strftime('%Y-%m-%d'),
                    'Confidence Score': confidence_score,
                    'Emotion Score': float(presentation_result.get('Emotion Score', 0)),
                    'Gesture Score': float(presentation_result.get('Gesture Score', 0)),
                    'Speech Sentiment Score': float(presentation_result.get('Vader Sentiment Score', 0)),
                    'Speech Rate Score': float(presentation_result.get('Speech Rate Score', 0)),
                    'Vocal Frequency Score': float(presentation_result.get('Frequency Score', 0)),
                    'Vocal Amplitude Score': float(presentation_result.get('Amplitude Score', 0)),
                    'Remarks': remarks
                }
                pdf_bytes = generate_pdf_report(pdf_data)
                
                if pdf_bytes:
                    report_id = fs.put(
                        pdf_bytes, 
                        filename=f"AutoCon_{analysis.get('file_name', 'report')}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                        user_id=analysis.get('user_id')
                    )
                    results_collection.update_one(
                        {'_id': ObjectId(analysis_id)},
                        {'$set': {'report_id': str(report_id)}}
                    )
                    return jsonify({
                        'message': 'Result updated successfully', 
                        'analysis_id': analysis_id,
                        'report_id': str(report_id)
                    }), 200
            
            return jsonify({'message': 'Result updated but PDF generation failed'}), 200
        else:
            video_path = request.args.get('video_path')
            if video_path and video_path in processed_videos:
                processed_videos[video_path].update({
                    'Confidence Score': confidence_score,
                    'Remarks': remarks,
                    'Emotion Score': emotion_score,
                    'Gesture Score': gesture_score,
                    'Vader Sentiment Score': sentiment_score,
                    'Speech Rate Score': speech_rate_score,
                    'Frequency Score': frequency_score,
                    'Amplitude Score': amplitude_score
                })
                
                return jsonify({'message': 'Result updated in processed videos'}), 200
            else:
                return jsonify({'error': 'No video path or analysis ID found'}), 400
    video_path = request.args.get('video_path')
    if video_path and video_path in processed_videos:
        presentation_result = processed_videos[video_path]
        gemini_result_analysis, gemini_recommendations = get_gemini_result_analysis(presentation_result)
        
        return render_template('result.html', 
                              presentation_result=presentation_result,
                              gemini_result_analysis=gemini_result_analysis,
                              gemini_recommendations=gemini_recommendations)
    analysis_id = session.get('processing_analysis_id')
    if analysis_id:
        analysis = results_collection.find_one({'_id': ObjectId(analysis_id)})
        if analysis and analysis.get('status') == 'completed':
            presentation_result = {
                'Confidence Score': analysis.get('confidence_score', 0),
                'Remarks': analysis.get('remarks', 'No remarks available'),
                'Emotion Score': analysis.get('emotion_score', 0),
                'Gesture Score': analysis.get('gesture_score', 0),
                'Vader Sentiment Score': analysis.get('speech_sentiment_score', 0),
                'Speech Rate Score': analysis.get('speech_rate_score', 0),
                'Frequency Score': analysis.get('vocal_frequency_score', 0),
                'Amplitude Score': analysis.get('vocal_amplitude_score', 0)
            }
            gemini_result_analysis, gemini_recommendations = get_gemini_result_analysis(presentation_result)
            
            return render_template('result.html', 
                                 presentation_result=presentation_result,
                                 gemini_result_analysis=gemini_result_analysis,
                                 gemini_recommendations=gemini_recommendations)    
    return redirect(url_for('upload'))
@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    user = session['user']
    
    user_id = user['id'] if isinstance(user, dict) and 'id' in user else str(user)
    
    user_history = []
    for result in results_collection.find({'user_id': user_id}).sort('timestamp', -1):
        user_history.append({
            'id': str(result['_id']),
            'timestamp': result['timestamp'],
            'file_name': result['file_name'],
            'confidence_score': result.get('confidence_score', 0),
            'remarks': result.get('remarks', 'No remarks available')
        })
    
    gemini_insights = get_gemini_analysis_insights(user_history)
    
    user_name = user['name'] if isinstance(user, dict) and 'name' in user else "User"
    
    return render_template('history.html', 
                          name=user_name, 
                          history=user_history,
                          gemini_insights=gemini_insights)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

def process_uploaded_video(video_path, selected_row_data):
    output_audio = 'Audio.wav'

    try:
        with processing_lock:
            selected_row_data = selected_row_data
            
            video_result = process_video(video_path)
            gesture_result = body_gesture(video_path)
            extract_audio_from_video(video_path=video_path, output_audio_path=output_audio)
            audio_result = process_audio('Audio.wav')

            video_result = json.loads(video_result)
            gesture_result = json.loads(gesture_result)
            audio_result = json.loads(audio_result)

            emotion = video_result['video_result']['dominant_emotion']
            gesture = gesture_result['gesture_result']['dominant_gesture']
            vader_sentiment = audio_result['final_vader_sentiment']
            speech_rate = audio_result['speech_rate']
            frequency = audio_result['frequency']
            amplitude = audio_result['average_amplitude']
            gender = selected_row_data['Gender'].lower()
            print(gender)

            # Evaluate presentation
            presentation_result = evaluate_presentation(emotion, vader_sentiment, speech_rate, frequency, gender, amplitude, gesture)
            
            # Ensure all scores use the same scale (0-100)
            for key in presentation_result:
                if key != 'Remarks' and isinstance(presentation_result[key], (int, float)):
                    # If the score is in 0-1 range, convert to 0-100 range
                    if presentation_result[key] <= 1.0:
                        presentation_result[key] = presentation_result[key] * 100

            print("Presentation Result",presentation_result)

            # Store the presentation result
            processed_videos[video_path] = presentation_result
            
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {e}")

    finally:
        # Remove the video from the processing set even if an exception occurs
        processing_videos.discard(video_path)

    return

def process_video_analysis(analysis_id, video_path):
    """Process a video for confidence analysis and update the database with results"""
    try:
        # Get the full path of the video
        full_path = os.path.join(os.getcwd(), video_path)
        
        # Extract audio from video
        output_audio = f'temp_audio_{analysis_id}.wav'
        extract_audio_from_video(video_path=full_path, output_audio_path=output_audio)
        
        # Process video for facial emotion analysis
        video_result = process_video(full_path) # Note: This is not JSON, returns dominant emotion directly
        
        # Process video for gesture analysis
        gesture_result = json.loads(body_gesture(full_path))
        
        # Process audio for sentiment and other audio metrics
        audio_result = json.loads(process_audio(output_audio))
        
        # Get the analysis record to retrieve the gender
        analysis = results_collection.find_one({'_id': ObjectId(analysis_id)})
        gender = analysis.get('gender', 'neutral').lower()
        
        # Extract the results with safer defaults
        emotion = video_result  # Just the dominant emotion, not nested in JSON
        emotion_score = 70  # Default value
        
        gesture = gesture_result['gesture_result']['dominant_gesture']
        gesture_score = gesture_result['gesture_result'].get('gesture_score', 70)
        
        vader_sentiment = audio_result['final_vader_sentiment']
        sentiment_score = audio_result.get('sentiment_score', 70)
        
        speech_rate = audio_result['speech_rate']
        speech_rate_score = audio_result.get('speech_rate_score', 70)
        
        frequency = audio_result['frequency']
        frequency_score = audio_result.get('frequency_score', 70)
        
        amplitude = audio_result['average_amplitude']
        amplitude_score = audio_result.get('amplitude_score', 70)
        
        # Evaluate the overall confidence
        presentation_result = evaluate_presentation(emotion, vader_sentiment, speech_rate, frequency, gender, amplitude, gesture)
        
        # Ensure all scores use the same scale (0-100)
        for key in presentation_result:
            if key != 'Remarks' and isinstance(presentation_result[key], (int, float)):
                # If the score is in 0-1 range, convert to 0-100 range
                if presentation_result[key] <= 1.0:
                    presentation_result[key] = presentation_result[key] * 100
        
        speech_rate_score = float(presentation_result.get('Speech Rate Score', 70))
        amplitude_score = float(presentation_result.get('Amplitude Score', 70))
        speech_clarity = round((speech_rate_score + amplitude_score) / 2, 1)
        
        # Calculate posture rating (based on gesture analysis)
        gesture_score = float(presentation_result.get('Gesture Score', 70))
        posture_rating = round(gesture_score * 1.05, 1) if gesture_score * 1.05 <= 100 else 99.5
        
        # Calculate engagement score (based on emotion and sentiment)
        emotion_score = float(presentation_result.get('Emotion Score', 70))
        sentiment_score = float(presentation_result.get('Vader Sentiment Score', 70))
        engagement_score = round((emotion_score + sentiment_score) / 2, 1)
        
        # Get the overall confidence score
        confidence_score = float(presentation_result.get('Confidence Score', 75))
        
        # Generate remarks based on the scores (without eye_contact parameter)
        remarks = generate_remarks_for_autocon(
            confidence_score, speech_clarity, 
            posture_rating, engagement_score, emotion, gesture
        )
        
        # Generate PDF report for the user
        pdf_data = {
            'User Name': analysis.get('user_name', 'User'),  # Get username directly from the analysis record
            'File Name': analysis['file_name'],
            'Confidence Score': confidence_score,
            'Remarks': remarks,
            'Evaluation Date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'Emotion Score': float(presentation_result.get('Emotion Score', 0)),
            'Gesture Score': float(presentation_result.get('Gesture Score', 0)),
            'Speech Sentiment Score': float(presentation_result.get('Vader Sentiment Score', 0)),
            'Speech Rate Score': float(presentation_result.get('Speech Rate Score', 0)),
            'Vocal Frequency Score': float(presentation_result.get('Frequency Score', 0)),
            'Vocal Amplitude Score': float(presentation_result.get('Amplitude Score', 0))
        }
        
        # Generate the PDF report
        pdf_bytes = generate_pdf_report(pdf_data)
        
        # Save the PDF to GridFS
        report_id = None
        if pdf_bytes:
            report_id = fs.put(
                pdf_bytes, 
                filename=f"AutoCon_{analysis['file_name']}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                user_id=analysis['user_id']
            )
        
        # Update the database with the analysis results
        results_collection.update_one(
            {'_id': ObjectId(analysis_id)},
            {'$set': {
                'status': 'completed',
                'confidence_score': confidence_score,
                'speech_clarity': speech_clarity,
                'posture_rating': posture_rating,
                'engagement_score': engagement_score,
                'emotion_score': float(presentation_result.get('Emotion Score', 0)),
                'gesture_score': float(presentation_result.get('Gesture Score', 0)),
                'speech_sentiment_score': float(presentation_result.get('Vader Sentiment Score', 0)),
                'speech_rate_score': float(presentation_result.get('Speech Rate Score', 0)),
                'vocal_frequency_score': float(presentation_result.get('Frequency Score', 0)),
                'vocal_amplitude_score': float(presentation_result.get('Amplitude Score', 0)),
                'remarks': remarks,
                'report_id': str(report_id) if report_id else None
            }}
        )
        if os.path.exists(output_audio):
            os.remove(output_audio)
            
    except Exception as e:
        # Update database with error status
        results_collection.update_one(
            {'_id': ObjectId(analysis_id)},
            {'$set': {
                'status': 'error',
                'error_message': str(e)
            }}
        )
        print(f"Error processing video {analysis_id}: {e}")
    
    return
@app.route('/check-result')
def check_result():
    video_path = request.args.get('video_path')
    if video_path in processed_videos:
        # Check if processing for this video is complete
        if video_path not in processing_videos:
            return jsonify({'result_available': True})
        else:
            return jsonify({'result_available': False, 'message': 'Video processing in progress'})
    else:
        return jsonify({'result_available': False, 'message': 'Video not processed'})
@app.route('/clear-videos', methods=['POST'])
def clear_videos():
    global processed_videos
    global processing_videos
    processed_videos = {}
    processing_videos.clear()

    return jsonify({'message': 'Videos data cleared successfully'}), 200
def get_progress_data(user_id):
    current_year = datetime.datetime.now().year
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_data = [None] * 12
    analyses_by_month = {}
    for analysis in results_collection.find({'user_id': user_id}):
        try:
            if 'timestamp' in analysis:
                ts = parser.parse(analysis['timestamp'])
                if ts.year == current_year:
                    month_idx = ts.month - 1
                    if month_idx not in analyses_by_month:
                        analyses_by_month[month_idx] = []
                    
                    analyses_by_month[month_idx].append(analysis)
        except Exception as e:
            print(f"Error parsing timestamp: {e}")
            continue
    for month_idx, month_analyses in analyses_by_month.items():
        if month_analyses:
            raw_avg = sum(a.get('confidence_score', 0) for a in month_analyses) / len(month_analyses)
            avg_confidence = round(raw_avg if raw_avg > 1 else raw_avg * 100, 1)
            monthly_data[month_idx] = avg_confidence
    return {
        'labels': month_names,
        'data': monthly_data
    }
def generate_remarks(confidence_score, eye_contact, speech_clarity, posture_rating, engagement_score, emotion, gesture):
    remarks = []
    if confidence_score >= 80:
        remarks.append("You demonstrate excellent overall confidence in your presentation.")
    elif confidence_score >= 65:
        remarks.append("You show good confidence in your presentation with room for improvement.")
    
    if eye_contact >= 80:
        remarks.append("Your eye contact is strong and engaging.")
    elif eye_contact >= 65:
        remarks.append("Your eye contact is decent but could be more consistent.")
    
    if speech_clarity >= 80:
        remarks.append("Your speech is very clear and well-paced.")
    elif speech_clarity >= 65:
        remarks.append("Your speech is generally clear but could benefit from more consistent pacing.")
    
    if posture_rating >= 80:
        remarks.append("Your posture and body language communicate confidence.")
    elif posture_rating >= 65:
        remarks.append("Your posture is generally good but could be more consistently confident.")
    
    if engagement_score >= 80:
        remarks.append("You effectively engage your audience throughout your presentation.")
    elif engagement_score >= 65:
        remarks.append("Your audience engagement is good but could be more consistent.")
    improvements = []
    
    if confidence_score < 65:
        improvements.append("Work on building your overall confidence through practice and preparation.")
    
    if eye_contact < 65:
        improvements.append("Improve your eye contact by consciously looking at your audience or camera.")
    
    if speech_clarity < 65:
        improvements.append("Focus on speaking more clearly and at a consistent pace.")
    
    if posture_rating < 65:
        improvements.append("Pay attention to your posture and body language to project more confidence.")
    
    if engagement_score < 65:
        improvements.append("Use more vocal variety and expressive gestures to engage your audience better.")
    
    # Add specific emotion and gesture feedback
    if emotion == "happy" or emotion == "surprised":
        remarks.append("Your positive facial expressions help connect with the audience.")
    elif emotion == "neutral":
        improvements.append("Try to incorporate more expressive facial reactions to better engage your audience.")
    elif emotion in ["sad", "angry", "fearful"]:
        improvements.append("Work on maintaining more positive facial expressions during your presentation.")
        
    if gesture == "confident":
        remarks.append("Your confident gestures enhance your overall presentation.")
    elif gesture == "neutral":
        improvements.append("Incorporate more purposeful hand gestures to emphasize key points.")
    elif gesture == "nervous":
        improvements.append("Try to reduce nervous gestures to appear more confident.")
    
    # Combine remarks and improvements
    final_remarks = " ".join(remarks)
    
    if improvements:
        final_remarks += " Areas for improvement: " + " ".join(improvements)
    
    return final_remarks

def generate_remarks_for_autocon(confidence_score, speech_clarity, posture_rating, engagement_score, emotion, gesture):
    """Generate personalized remarks based on the analysis results without eye contact parameter"""
    remarks = []
    
    # Add positive remarks
    if confidence_score >= 80:
        remarks.append("You demonstrate excellent overall confidence in your presentation.")
    elif confidence_score >= 65:
        remarks.append("You show good confidence in your presentation with room for improvement.")
    
    if speech_clarity >= 80:
        remarks.append("Your speech is very clear and well-paced.")
    elif speech_clarity >= 65:
        remarks.append("Your speech is generally clear but could benefit from more consistent pacing.")
    
    if posture_rating >= 80:
        remarks.append("Your posture and body language communicate confidence.")
    elif posture_rating >= 65:
        remarks.append("Your posture is generally good but could be more consistently confident.")
    
    if engagement_score >= 80:
        remarks.append("You effectively engage your audience throughout your presentation.")
    elif engagement_score >= 65:
        remarks.append("Your audience engagement is good but could be more consistent.")
    
    # Add areas for improvement
    improvements = []
    
    if confidence_score < 65:
        improvements.append("Work on building your overall confidence through practice and preparation.")
    
    if speech_clarity < 65:
        improvements.append("Focus on speaking more clearly and at a consistent pace.")
    
    if posture_rating < 65:
        improvements.append("Pay attention to your posture and body language to project more confidence.")
    
    if engagement_score < 65:
        improvements.append("Use more vocal variety and expressive gestures to engage your audience better.")
    
    # Add specific emotion and gesture feedback
    if emotion == "happy" or emotion == "surprised":
        remarks.append("Your positive facial expressions help connect with the audience.")
    elif emotion == "neutral":
        improvements.append("Try to incorporate more expressive facial reactions to better engage your audience.")
    elif emotion in ["sad", "angry", "fearful"]:
        improvements.append("Work on maintaining more positive facial expressions during your presentation.")
        
    if gesture == "confident":
        remarks.append("Your confident gestures enhance your overall presentation.")
    elif gesture == "neutral":
        improvements.append("Incorporate more purposeful hand gestures to emphasize key points.")
    elif gesture == "nervous":
        improvements.append("Try to reduce nervous gestures to appear more confident.")
    
    final_remarks = " ".join(remarks)
    
    if improvements:
        final_remarks += " Areas for improvement: " + " ".join(improvements)
    
    return final_remarks

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == "__main__":
    threading.Timer(1, open_browser).start()  # Open the browser after a short delay
    app.run()
