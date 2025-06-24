from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO

def generate_pdf_report(user_data):
    # Extract user information
    name = user_data.get('User Name', 'User')
    email = user_data.get('Email', '')
    mobile_number = user_data.get('Mobile Number', '')
    file_name = user_data.get('File Name', '')
    confidence_score = user_data.get('Confidence Score', '')
    remarks = user_data.get('Remarks', '')
    evaluation_date = user_data.get('Evaluation Date', '')
    emotion_score = user_data.get('Emotion Score', '')
    gesture_score = user_data.get('Gesture Score', '')
    vader_sentiment_score = user_data.get('Speech Sentiment Score', '')
    speech_rate_score = user_data.get('Speech Rate Score', '')
    frequency_score = user_data.get('Vocal Frequency Score', '')
    amplitude_score = user_data.get('Vocal Amplitude Score', '')

    # Create a BytesIO object to store PDF content in memory
    pdf_buffer = BytesIO()

    # Create a new PDF document
    c = canvas.Canvas(pdf_buffer, pagesize=letter)

    # Set page margins and spacing
    top_margin = 50
    left_margin = 50
    line_height = 20

    # Load logo and get its dimensions
    logo = ImageReader('static/logo.jpg')
    logo_width = 80
    logo_height = 80

    # Draw the logo at the top left with spacing
    c.drawImage(logo, left_margin, letter[1] - top_margin - logo_height + 18 , width=logo_width, height=logo_height, preserveAspectRatio=True)

    # Calculate the starting x-coordinate for the heading
    heading_x = left_margin + logo_width + 35  # Add spacing between logo and heading

    # Draw the heading with underline
    c.setFont("Helvetica-Bold", 20)
    c.drawString(heading_x, letter[1] - top_margin - 20, "AutoCon Confidence Analysis Report")
    c.line(heading_x, letter[1] - top_margin - 25, heading_x + c.stringWidth("AutoCon Confidence Analysis Report", "Helvetica-Bold", 20), letter[1] - top_margin - 25)

    # Define content for the PDF
    content = [
        ("User Name", name),
        ("Email", email),
        ("File Analyzed", file_name),
        ("Evaluation Date", evaluation_date),
        ("Overall Confidence Score", f"{confidence_score}%"),
    ]

    # Write content to the PDF
    current_y = letter[1] - top_margin - logo_height - 70  # Start below logo
    for parameter, value in content:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left_margin, current_y, f"{parameter}:")
        c.setFont("Helvetica", 12)
        c.drawString(left_margin + 150, current_y, str(value))
        current_y -= line_height  # Move to the next line
    
    # Add Remarks section
    current_y -= 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(left_margin, current_y, "Analysis Remarks:")
    current_y -= line_height
    
    # Handle wrapping for longer remarks text
    c.setFont("Helvetica", 12)
    remarks_width = letter[0] - 2 * left_margin
    remarks_text = c.beginText(left_margin, current_y)
    remarks_text.setFont("Helvetica", 12)
    
    # Split remarks into lines that fit the page width
    words = remarks.split()
    line = ""
    for word in words:
        test_line = line + " " + word if line else word
        if c.stringWidth(test_line, "Helvetica", 12) < remarks_width:
            line = test_line
        else:
            remarks_text.textLine(line)
            current_y -= line_height
            line = word
    if line:  # Add the last line
        remarks_text.textLine(line)
    
    c.drawText(remarks_text)

    # Add Individual Scores section
    c.setFont("Helvetica-Bold", 16)
    current_y -= 40  # Add space before Individual Scores
    c.drawString(left_margin, current_y, "Individual Metrics Analysis")

    individual_scores = [
        ("Facial Emotion Score", f"{emotion_score}%"),
        ("Gesture Analysis Score", f"{gesture_score}%"),
        ("Speech Sentiment Score", f"{vader_sentiment_score}%"),
        ("Speech Rate Score", f"{speech_rate_score}%"),
        ("Vocal Frequency Score", f"{frequency_score}%"),
        ("Vocal Amplitude Score", f"{amplitude_score}%")
    ]

    current_y -= line_height+10  # Move down for scores
    for parameter, value in individual_scores:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left_margin, current_y, f"{parameter}:")
        c.setFont("Helvetica", 12)
        c.drawString(left_margin + 170, current_y, str(value))
        current_y -= line_height  # Move to the next line

    # Add footer with date and page number
    c.setFont("Helvetica", 9)
    c.drawString(left_margin, 30, f"Generated on {evaluation_date} by AutoCon AI")
    c.drawRightString(letter[0] - left_margin, 30, "Page 1")

    # Save the PDF document content into the BytesIO buffer
    c.save()

    # Reset the buffer position to the beginning
    pdf_buffer.seek(0)

    # Return the PDF content as bytes
    return pdf_buffer.getvalue()

