import random

def evaluate_presentation(emotion, vader_sentiment, speech_rate, frequency, gender, amplitude, gesture):
    def combine_outputs(emotion, vader_sentiment, speech_rate, frequency, gender, amplitude, gesture):
        # Normalize input values to handle case variations and conversions
        emotion = emotion.lower() if emotion else "neutral"
        vader_sentiment = vader_sentiment.lower() if vader_sentiment else "neutral"
        speech_rate = speech_rate if speech_rate else "Moderate"
        gesture = gesture if gesture else "Neutral Pose"
        
        # Convert happiness to happy if needed (mapping between detected emotion and scoring)
        if emotion == "happy":
            emotion = "happiness"
        
        # Safe score calculations with defaults
        emotion_score_val = emotion_score(emotion)
        vader_sentiment_score_val = vader_sentiment_score(vader_sentiment)
        speech_rate_score_val = speech_rate_score(speech_rate)
        
        try:
            frequency_score_val = vocal_frequency_score(frequency, gender)
        except ValueError:
            # Default to a middle score if gender is invalid
            frequency_score_val = 0.65
            
        amplitude_score_val = amplitude_score(amplitude)
        gesture_score_val = gesture_score(gesture)
        
        # Ensure all values are numeric before addition
        emotion_score_val = 0.7 if emotion_score_val is None else emotion_score_val
        vader_sentiment_score_val = 0.7 if vader_sentiment_score_val is None else vader_sentiment_score_val
        speech_rate_score_val = 0.7 if speech_rate_score_val is None else speech_rate_score_val
        frequency_score_val = 0.7 if frequency_score_val is None else frequency_score_val
        amplitude_score_val = 0.7 if amplitude_score_val is None else amplitude_score_val
        gesture_score_val = 0.7 if gesture_score_val is None else gesture_score_val
        
        # Calculate combined score with safe values
        combined_score = (emotion_score_val + vader_sentiment_score_val + speech_rate_score_val +
                         frequency_score_val + amplitude_score_val + gesture_score_val) / 6
        
        # Collect all scores in a dictionary
        scores = {
            'Emotion Score': emotion_score_val,
            'Vader Sentiment Score': vader_sentiment_score_val,
            'Speech Rate Score': speech_rate_score_val,
            'Frequency Score': frequency_score_val,
            'Amplitude Score': amplitude_score_val,
            'Gesture Score': gesture_score_val,
            'Combined Score': combined_score
        }
        
        return scores
    
    def assign_score(combined_score):
        # Assign a score on a 10-point scale based on the combined score
        score = int(round(combined_score * 10))
        return score

    def emotion_score(emotion):
        # Assign scores to emotions based on the specified order
        emotion_scores = {'happiness': random.uniform(0.8, 1.0), 'neutral': random.uniform(0.7, 0.9), 
                          'surprise': random.uniform(0.6, 0.7), 'fear': random.uniform(0.5, 0.6), 
                          'sadness': random.uniform(0.4, 0.5), 'anger': random.uniform(0.2, 0.4), 
                          'disgust': random.uniform(0.1, 0.3)}
        # Return a default score if emotion not found
        return emotion_scores.get(emotion, random.uniform(0.5, 0.7))
    
    def gesture_score(gesture):
        # Assign scores based on the gesture condition
        gesture_scores = {
            'Open Pose': random.uniform(0.8, 1.0),
            'Relaxed Pose': random.uniform(0.7, 0.9),
            'In Control Pose': random.uniform(0.7, 1.0),
            'Closed Pose': random.uniform(0.4, 0.6)
        }
        return gesture_scores.get(gesture, random.uniform(0.5, 0.7))

    def vader_sentiment_score(vader_sentiment):
        # Assign scores to VADER sentiment with case insensitive matching
        if vader_sentiment.lower() in ['positive', 'pos']:
            return random.uniform(0.7, 1.0)
        elif vader_sentiment.lower() in ['negative', 'neg']:
            return random.uniform(0.2, 0.6)
        else:
            return random.uniform(0.6, 0.8) # For Neutral

    def speech_rate_score(speech_rate):
        # Assign scores to speech rate categories
        speech_rate_scores = {'Very Fast': random.uniform(0.6, 0.8), 'Fast': random.uniform(0.7, 0.9), 
                             'Moderate': random.uniform(0.7, 1.0), 'Slow': random.uniform(0.4, 0.6), 
                             'Very Slow': random.uniform(0.2, 0.4)}
        # Return a default score if speech rate not found
        return speech_rate_scores.get(speech_rate, random.uniform(0.5, 0.7))

    def vocal_frequency_score(frequency, gender):
        # Handle None frequency
        if frequency is None:
            return 0.65  # Default middle score
            
        # Make gender case-insensitive
        gender = gender.lower() if isinstance(gender, str) else "neutral"
        
        # Define the standard vocal frequency ranges for male and female speakers
        male_frequency_range = [(85, 180), (165, 255)]  # (lower range, higher range)
        female_frequency_range = [(165, 255), (250, 500)]  # (lower range, higher range)
        neutral_frequency_range = [(120, 400)]  # For neutral/other gender options

        # Determine the appropriate frequency ranges based on gender
        if gender == 'male':
            frequency_ranges = male_frequency_range
        elif gender == 'female':
            frequency_ranges = female_frequency_range
        else:
            # Use neutral range instead of raising error
            frequency_ranges = neutral_frequency_range

        # Check if the frequency falls within any of the standard ranges
        for freq_range in frequency_ranges:
            if frequency >= freq_range[0] and frequency <= freq_range[1]:
                # If within the range, assign a higher score
                return random.uniform(0.7, 1.0)

        # If outside all standard ranges, assign a lower score
        return random.uniform(0.6, 0.7)

    def amplitude_score(amplitude):
        # Handle None amplitude
        if amplitude is None:
            return 0.6  # Default score
            
        # Define thresholds for amplitude ranges
        low_threshold = 0.005
        high_threshold = 0.02

        # Assign scores based on amplitude ranges
        if amplitude < low_threshold:
            return random.uniform(0.2, 0.4)
        elif low_threshold <= amplitude < high_threshold:
            return random.uniform(0.5, 0.7)
        else:
            return random.uniform(0.6, 0.8)

    # Combine outputs to get individual scores and combined score
    scores = combine_outputs(emotion, vader_sentiment, speech_rate, frequency, gender, amplitude, gesture)
    
    # Assign a score based on the combined score
    score = assign_score(scores['Combined Score'])

    # Convert score to percentage (0-100 scale)
    percentage_score = min(score * 10, 100)  # Cap at 100
    
    # Determine remarks based on confidence score
    if score >= 8:
        remarks = "Excellent Presentation Skills"
    elif score >= 7 and score < 8:
        remarks = "CONFIDENT.....WELL DONE"
    elif score >= 6 and score < 7:
        remarks = "Good Presentation.....Keep working on it."
    elif score >= 4 and score < 6:
        remarks = "NOT UP TO THE MARK.....NEEDS IMPROVEMENT"
    else:
        remarks = "LACK OF CONFIDENCE.....NEEDS RE-EVALUATION"

    # Add confidence score and remarks to the scores dictionary
    scores.update({
        'Confidence Score': percentage_score,  # Use percentage score
        'Remarks': remarks
    })

    return scores

