# main.py - AI/ML Internship Email Automation System (TXT Resume, Clean Output)
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, FileReadTool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from crewai import LLM
from pathlib import Path

# Load environment variables
load_dotenv()

class EmailAutomationCrew:
    def __init__(self):
        # Initialize Gemini LLM
        self.llm = LLM(
            model="gemini/gemini-2.0-flash",
            temperature=0.7,
        )
        
        # Initialize tools
        self.search_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))
        self.file_read_tool = FileReadTool()
        
        # Create agents
        self.research_agent = self._create_research_agent()
        self.analysis_agent = self._create_analysis_agent()
        self.resume_agent = self._create_resume_agent()
        self.editor_agent = self._create_editor_agent()
        self.auditor_agent = self._create_auditor_agent()

    def _create_research_agent(self):
        return Agent(
            role='Company Research Specialist',
            goal='Research comprehensive information about companies for AI/ML internship opportunities',
            backstory="""You are an expert researcher who specializes in gathering detailed 
            information about companies, their culture, recent developments, AI/ML initiatives, 
            and hiring practices. You excel at finding relevant and up-to-date information 
            that can be used to craft personalized job applications.""",
            verbose=False,
            allow_delegation=False,
            tools=[self.search_tool],
            llm=self.llm
        )

    def _create_analysis_agent(self):
        return Agent(
            role='Company Analysis Expert',
            goal='Analyze company information and draft initial email content for AI/ML internship applications',
            backstory="""You are a strategic analyst who excels at understanding company 
            dynamics, culture, and requirements. You can identify key points that make a 
            candidate attractive for AI/ML internship positions and craft compelling initial 
            email drafts that align with company values and needs.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )

    def _create_resume_agent(self):
        return Agent(
            role='Resume Information Extractor and Matcher',
            goal='Extract relevant information from TXT resume and intelligently match it with company requirements',
            backstory="""You are a professional resume analyst who specializes in parsing 
            and analyzing resumes from TXT files. You excel at identifying the most relevant 
            skills, experiences, projects, and achievements that match specific company 
            requirements in the AI/ML field. You understand how to extract key information 
            like technical skills, programming languages, frameworks, projects, education, 
            certifications, and work experience, then strategically align them with company needs.""",
            verbose=False,
            allow_delegation=False,
            tools=[self.file_read_tool],
            llm=self.llm
        )

    def _create_editor_agent(self):
        return Agent(
            role='Professional Email Editor',
            goal='Create polished, professional emails for AI/ML internship applications',
            backstory="""You are a professional communication expert who specializes in 
            crafting compelling job application emails. You understand the nuances of 
            professional communication and can create emails that are engaging, respectful, 
            and gender-neutral while highlighting the candidate's strengths.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )

    def _create_auditor_agent(self):
        return Agent(
            role='Email Quality Auditor',
            goal='Review and perfect email content for final submission',
            backstory="""You are a meticulous quality assurance specialist who ensures 
            all communications are error-free, professional, and optimized for success. 
            You have an eye for detail and can catch any grammatical errors, formatting 
            issues, or content problems that might affect the email's effectiveness.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )

    def _extract_resume_content(self, resume_path):
        """
        Extract content from TXT resume file
        """
        try:
            file_path = Path(resume_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Resume file not found: {resume_path}")
            
            file_extension = file_path.suffix.lower()
            
            if file_extension != '.txt':
                raise ValueError(f"Only TXT files are supported. Found: {file_extension}")
            
            with open(resume_path, 'r', encoding='utf-8') as file:
                return file.read()
                
        except Exception as e:
            return f"Error reading resume file: {str(e)}"

    def create_tasks(self, company_name, person_name, person_email, resume_path="resume.txt"):
        # Task 1: Research Company
        research_task = Task(
            description=f"""Research comprehensive information about {company_name}. 
            Find details about:
            - Company overview and mission
            - Recent news and developments
            - AI/ML initiatives and projects
            - Company culture and values
            - Internship programs and hiring practices
            - Key technologies they use (Python, TensorFlow, PyTorch, etc.)
            - Recent achievements or milestones
            - Preferred skills for AI/ML roles
            - Company size and work environment
            
            Provide a detailed report with all relevant information that can be used to customize the application.""",
            agent=self.research_agent,
            expected_output="A comprehensive research report about the company including their AI/ML focus, culture, recent developments, internship opportunities, and preferred candidate profile."
        )

        # Task 2: Analyze Company and Draft Initial Email Strategy
        analysis_task = Task(
            description=f"""Based on the research findings about {company_name}, analyze the 
            company's needs and culture, then create a strategic framework for the email. Focus on:
            - Key points that would appeal to this specific company
            - How AI/ML skills should be positioned for their needs
            - Company-specific customization points
            - Appropriate tone and approach
            - What type of projects/experience they would value most
            - Technical skills they prioritize
            
            Create a strategic email framework that highlights what matters most to this company.""",
            agent=self.analysis_agent,
            expected_output="A strategic email framework tailored to the company's specific needs, culture, and AI/ML requirements, with key points for personalization.",
            context=[research_task]
        )

        # Task 3: Extract and Match Resume Information
        resume_task = Task(
            description=f"""Extract and analyze resume information from the TXT file at {resume_path}. 
            Based on the company analysis for {company_name}, identify and organize:
            
            TECHNICAL SKILLS:
            - Programming languages (Python, R, Java, etc.)
            - AI/ML frameworks (TensorFlow, PyTorch, Scikit-learn, etc.)
            - Data tools and libraries (Pandas, NumPy, Matplotlib, etc.)
            - Cloud platforms (AWS, GCP, Azure)
            - Databases and big data tools
            
            RELEVANT EXPERIENCE:
            - AI/ML projects (personal, academic, or professional)
            - Data science or analytics experience
            - Software development experience
            - Research experience
            - Internships or work experience
            
            EDUCATION:
            - Degree and field of study
            - Relevant coursework
            - Academic projects
            - GPA (if strong)
            
            ACHIEVEMENTS:
            - Certifications
            - Publications or research
            - Competition wins
            - Notable accomplishments
            
            Prioritize and match these elements with {company_name}'s specific requirements and interests.
            Extract the actual resume content first, then analyze and categorize it.""",
            agent=self.resume_agent,
            expected_output="A comprehensive analysis of the candidate's qualifications extracted from the TXT resume, organized by relevance to the company's AI/ML internship requirements, with specific examples and achievements highlighted.",
            context=[research_task, analysis_task]
        )

        # Task 4: Create Professional Email
        editor_task = Task(
            description=f"""Using the company research, strategic framework, and resume analysis, 
            create a professional, compelling email for an AI/ML internship application to 
            {person_name} at {company_name}. The email should:
            
            STRUCTURE:
            - Compelling subject line
            - Professional greeting
            - Strong opening that mentions specific company interest
            - 2-3 body paragraphs highlighting relevant qualifications
            - Clear call to action
            - Professional closing
            
            CONTENT REQUIREMENTS:
            - Be gender-neutral and professional
            - Incorporate specific company insights and show research
            - Highlight most relevant technical skills naturally
            - Include 1-2 specific projects or experiences
            - Show genuine interest and cultural fit
            - Demonstrate understanding of their AI/ML work
            - Keep it concise but comprehensive (300-400 words)
            
            PERSONALIZATION:
            - Reference specific company projects or initiatives
            - Align candidate's experience with company needs
            - Use appropriate technical terminology
            - Show enthusiasm for their specific work
            
            Recipient: {person_name}
            Email: {person_email}
            Company: {company_name}""",
            agent=self.editor_agent,
            expected_output="A complete, professional email ready for sending, with proper formatting, compelling content, and strong alignment between candidate qualifications and company needs.",
            context=[research_task, analysis_task, resume_task]
        )

        # Task 5: Final Review and Audit
        auditor_task = Task(
            description=f"""Perform a comprehensive final review of the email for {person_name} at {company_name}. 
            
            CHECK FOR:
            - Grammar, spelling, and punctuation errors
            - Professional tone and appropriate formality
            - Clarity and coherence of message
            - Appropriate length (not too long or short)
            - All necessary information included
            - Gender-neutral language throughout
            - Proper email structure and formatting
            - Contact information accuracy
            - Technical terminology used correctly
            - Company name and person name spelled correctly
            - Strong subject line included
            
            VERIFY ALIGNMENT:
            - Resume information accurately represented
            - Company research properly incorporated
            - Technical skills appropriately highlighted
            - Projects and experience relevantly positioned
            
            Personal Info for contact:
            - Name: Avinash Tiwari
            - mobile: +91 8299661089
            - linkedin: https://www.linkedin.com/in/avinash-tiwari-bba572278/
            - github: https://github.com/avinash4002
            
            
            Use only these personal details for contact information.
            Don't mention my email or any other personal details in the email.
            
            IMPORTANT: Return ONLY the final email content. Do not include any explanations, 
            analysis, or additional text. The output should be the clean, ready-to-send email only.""",
            agent=self.auditor_agent,
            expected_output="ONLY the final email content - no additional text, explanations, or formatting. Just the clean email ready for sending.",
            context=[research_task, analysis_task, resume_task, editor_task]
        )

        return [research_task, analysis_task, resume_task, editor_task, auditor_task]

    def run(self, company_name, person_name, person_email, resume_path="resume.txt"):
        # Validate resume file first
        resume_content = self._extract_resume_content(resume_path)
        if resume_content.startswith("Error"):
            print(f"❌ {resume_content}")
            return None
        
        # Create tasks
        tasks = self.create_tasks(company_name, person_name, person_email, resume_path)
        
        # Create crew
        crew = Crew(
            agents=[
                self.research_agent,
                self.analysis_agent,
                self.resume_agent,
                self.editor_agent,
                self.auditor_agent
            ],
            tasks=tasks,
            verbose=False,
            process=Process.sequential
        )
        
        # Execute the crew
        result = crew.kickoff()
        return result

def main():
    # Input parameters
    company_name = input("Enter company name: ").strip()
    person_name = input("Enter HR person's name: ").strip()
    person_email = input("Enter HR person's email: ").strip()
    resume_path = input("Enter resume file path (default: resume.txt): ").strip() or "resume.txt"
    
    if not all([company_name, person_name, person_email]):
        print("❌ Error: All fields are required!")
        return
    
    try:
        email_crew = EmailAutomationCrew()
        result = email_crew.run(company_name, person_name, person_email, resume_path)
        
        if result:
            # Print only the email content
            print(result)
            
            # Save result to file
            output_filename = f"email_{company_name.replace(' ', '_').lower()}.txt"
            with open(output_filename, "w", encoding='utf-8') as f:
                f.write(str(result))
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()