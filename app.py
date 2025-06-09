import csv
from email_generator import generate_email
from email_sender import send_email_with_resume
import os
from dotenv import load_dotenv

# Load environment variables (for password)
load_dotenv()

# Parameters
SENDER_EMAIL = "avinashtiwari1089@gmail.com"
RESUME_FILE = "avinash_resume.txt"  # Make sure this file exists in your project folder
CSV_FILE = "output.csv"  # Your CSV with Name,Email,Company

def send_bulk_emails():
    try:
        with open(CSV_FILE, newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                hr_name = row["Name"]
                hr_email = row["Email"]
                company = row["Company"]

                print(f"\n📧 Generating email for {hr_name} at {company}...")

                email_body = generate_email(
                    company,
                    hr_name,
                    hr_email,
                    RESUME_FILE
                )

                if not email_body:
                    print(f"❌ Failed to generate email for {hr_email}. Skipping.")
                    continue

                subject = f"Application for a Intern Role at {company}"

                send_email_with_resume(
                    sender_email=SENDER_EMAIL,
                    receiver_email=hr_email,
                    subject=subject,
                    email_body=email_body,
                    resume_path="Avinash(exp).pdf"
                )
    except FileNotFoundError:
        print(f"❌ CSV file '{CSV_FILE}' not found.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    send_bulk_emails()
