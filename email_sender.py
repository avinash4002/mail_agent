import smtplib
from email.message import EmailMessage
import os
from dotenv import load_dotenv
import mimetypes

load_dotenv()

def send_email_with_resume(sender_email, receiver_email, subject, email_body, resume_path):
    # Ensure email_body is a plain string
    if not isinstance(email_body, str):
        email_body = str(email_body)

    # Compose the email
    msg = EmailMessage()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.set_content(email_body)

    # Attach the resume
    try:
        with open(resume_path, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(resume_path)
            mime_type, _ = mimetypes.guess_type(resume_path)
            main_type, sub_type = mime_type.split('/', 1)

            msg.add_attachment(file_data, maintype=main_type, subtype=sub_type, filename=file_name)
    except FileNotFoundError:
        print(f"❌ Resume file not found: {resume_path}")
        return

    # Send the email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.starttls()
            smtp.login(sender_email, os.getenv("EMAIL_PASSWORD"))  # App password recommended
            smtp.send_message(msg)
            print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Error sending email: {e}")
