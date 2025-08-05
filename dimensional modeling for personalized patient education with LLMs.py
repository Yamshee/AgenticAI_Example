#This code example demonstrates how you can implement a dimensional model in Python using pandas DataFrames and then leverage this structure with LangChain and an LLM to 
#generate personalized patient education materials.

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Set your OpenAI API key as an environment variable or replace with your key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# --- 1. Simulate the dimensional model with pandas DataFrames ---

# Fact Table: Patient Encounters
fact_patient_encounter_data = {
    'encounter_id': [101, 102, 103, 104, 105],
    'patient_key': [1, 2, 3, 1, 4],
    'diagnosis_key': [1001, 1002, 1003, 1001, 1002],
    'provider_key': [201, 202, 203, 201, 202],
    'date_key': [301, 302, 303, 304, 305],
    'treatment_plan_key': [401, 402, 403, 401, 402],
    'readmission_flag': [False, False, True, False, False],
    'patient_satisfaction_score': [9, 8, 6, 9, 7]
}
fact_patient_encounter = pd.DataFrame(fact_patient_encounter_data)

# Dimension Table: Patients
dim_patient_data = {
    'patient_key': [1, 2, 3, 4, 5],
    'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
    'first_name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'last_name': ['Smith', 'Johnson', 'Williams', 'Brown', 'Davis'],
    'age_group': ['65+', '36-50', '75+', '65+', '51-65'],
    'gender': ['Female', 'Male', 'Male', 'Male', 'Female'],
    'ethnicity': ['Caucasian', 'African American', 'Caucasian', 'Hispanic', 'Asian'],
    'primary_language': ['English', 'English', 'Spanish', 'English', 'Mandarin'],
    'insurance_type': ['Medicare', 'Commercial', 'Medicare', 'Medicaid', 'Commercial'],
    'lifestyle_factors': ['Sedentary', 'Active', 'Moderate', 'Sedentary', 'Active'] # Simplified
}
dim_patient = pd.DataFrame(dim_patient_data)

# Dimension Table: Diagnoses
dim_diagnosis_data = {
    'diagnosis_key': [1001, 1002, 1003],
    'icd_code': ['E11.9', 'I10', 'I50.9'],
    'diagnosis_name': ['Type 2 Diabetes Mellitus', 'Essential (primary) hypertension', 'Heart Failure, unspecified'],
    'diagnosis_category': ['Endocrine Disorders', 'Circulatory System Diseases', 'Circulatory System Diseases'],
    'severity': ['Moderate', 'Moderate', 'Severe']
}
dim_diagnosis = pd.DataFrame(dim_diagnosis_data)

# Dimension Table: Providers
dim_provider_data = {
    'provider_key': [201, 202, 203],
    'provider_id': ['DR001', 'DR002', 'DR003'],
    'provider_name': ['Dr. Emily White', 'Dr. Michael Green', 'Dr. Sarah Black'],
    'specialty': ['Endocrinologist', 'Cardiologist', 'General Practitioner'],
    'hospital_affiliation': ['City Hospital', 'Community Clinic', 'City Hospital']
}
dim_provider = pd.DataFrame(dim_provider_data)

# Dimension Table: Dates
dim_date_data = {
    'date_key': [301, 302, 303, 304, 305],
    'full_date': pd.to_datetime(['2025-01-15', '2025-01-20', '2025-01-25', '2025-02-01', '2025-02-05']),
    'year': [2025, 2025, 2025, 2025, 2025],
    'month': [1, 1, 1, 2, 2],
    'day_of_week': ['Wednesday', 'Monday', 'Saturday', 'Saturday', 'Wednesday']
}
dim_date = pd.DataFrame(dim_date_data)

# Dimension Table: Treatment Plans
dim_treatment_plan_data = {
    'treatment_plan_key': [401, 402, 403],
    'medications': ['Metformin, Lisinopril', 'Hydrochlorothiazide', 'Furosemide'],
    'follow_up_frequency': ['Monthly', 'Quarterly', 'Bi-weekly'],
    'lifestyle_recommendations': ['Low sugar diet, Regular exercise', 'Low sodium diet', 'Fluid restriction, Low sodium diet']
}
dim_treatment_plan = pd.DataFrame(dim_treatment_plan_data)

# --- 2. Function to retrieve patient data and generate prompt ---

def get_patient_data_and_prompt(patient_id: str) -> dict:
    """
    Retrieves patient data from the dimensional model and generates a prompt
    for the LLM based on the patient's information.
    """
    patient_info = dim_patient[dim_patient['patient_id'] == patient_id].iloc[0]
    patient_key = patient_info['patient_key']

    # Join with fact table to get most recent encounter details
    latest_encounter = fact_patient_encounter[fact_patient_encounter['patient_key'] == patient_key].sort_values(by='date_key', ascending=False).iloc[0]

    diagnosis_info = dim_diagnosis[dim_diagnosis['diagnosis_key'] == latest_encounter['diagnosis_key']].iloc[0]
    treatment_info = dim_treatment_plan[dim_treatment_plan['treatment_plan_key'] == latest_encounter['treatment_plan_key']].iloc[0]

    # Construct the context for the LLM
    context = {
        "patient_name": f"{patient_info['first_name']} {patient_info['last_name']}",
        "age_group": patient_info['age_group'],
        "primary_language": patient_info['primary_language'],
        "diagnosis_name": diagnosis_info['diagnosis_name'],
        "diagnosis_category": diagnosis_info['diagnosis_category'],
        "medications": treatment_info['medications'],
        "lifestyle_recommendations": treatment_info['lifestyle_recommendations'],
        "severity": diagnosis_info['severity']
    }

    # Generate the prompt using LangChain's ChatPromptTemplate
    template_str = """
    You are a helpful healthcare assistant.
    Generate personalized patient education material based on the following information:
    Patient Name: {patient_name}
    Age Group: {age_group}
    Primary Language: {primary_language}
    Diagnosis: {diagnosis_name} ({diagnosis_category}, Severity: {severity})
    Medications: {medications}
    Lifestyle Recommendations: {lifestyle_recommendations}

    Explain the diagnosis in simple terms, considering the patient's age group and language.
    Clearly explain the purpose and importance of the prescribed medications.
    Provide actionable advice for the recommended lifestyle changes.
    Emphasize the importance of adherence to the treatment plan.
    Keep the explanation concise and easy to understand.
    """

    prompt = ChatPromptTemplate.from_template(template_str)
    formatted_prompt = prompt.format(**context)

    return {"context": context, "prompt": formatted_prompt}

# --- 3. Initialize LLM and Chain for text generation ---

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
output_parser = StrOutputParser() # To get a plain string output from the LLM

# Define a simple chain: prompt -> llm -> output_parser
chain = (
    ChatPromptTemplate.from_template("{prompt}")
    | llm
    | output_parser
)

# --- 4. Example usage ---

if __name__ == "__main__":
    patient_id_to_educate = 'P001' # Example patient ID

    patient_data = get_patient_data_and_prompt(patient_id_to_educate)
    print(f"--- Patient Context for {patient_id_to_educate} ---")
    for key, value in patient_data['context'].items():
        print(f"{key}: {value}")
    print("\n")

    print("--- Generated Prompt for LLM ---")
    print(patient_data['prompt'])
    print("\n")

    print("--- LLM Generated Patient Education Material ---")
    education_material = chain.invoke({"prompt": patient_data['prompt']})
    print(education_material)

    print("\n" + "="*50 + "\n")

    patient_id_to_educate = 'P003' # Another example patient ID
    patient_data = get_patient_data_and_prompt(patient_id_to_educate)
    print(f"--- Patient Context for {patient_id_to_educate} ---")
    for key, value in patient_data['context'].items():
        print(f"{key}: {value}")
    print("\n")

    print("--- Generated Prompt for LLM ---")
    print(patient_data['prompt'])
    print("\n")

    print("--- LLM Generated Patient Education Material ---")
    education_material = chain.invoke({"prompt": patient_data['prompt']})
    print(education_material)



"""

How the code demonstrates dimensional modeling and LLM integration
Dimensional Model Implementation:
Separate pandas DataFrames are used to simulate the fact table (fact_patient_encounter) and dimension tables (dim_patient, dim_diagnosis, dim_provider, dim_date, dim_treatment_plan).
Each DataFrame has a primary key (_key) and foreign keys to link to other tables, reflecting the star schema approach.
This structure allows for easy retrieval of specific information by joining or filtering across these tables, mirroring how a dimensional model facilitates data access in a data warehouse.
Data Extraction and Feature Engineering:
The get_patient_data_and_prompt function demonstrates how to extract specific features for a given patient from the dimensional model.
It joins information from the dim_patient, fact_patient_encounter, dim_diagnosis, and dim_treatment_plan tables to gather a comprehensive context for the LLM.
These extracted pieces of information (patient_name, age_group, diagnosis_name, medications, etc.) act as features that will personalize the LLM's output.
LLM Integration with LangChain:
Prompt Engineering: A ChatPromptTemplate is used to define the structure of the prompt. This template includes placeholders for the extracted patient features, allowing for dynamic and personalized prompts.
LLM Model: A ChatOpenAI instance is initialized, representing the large language model.
Output Parsing: StrOutputParser is used to ensure the LLM's response is returned as a simple string.
Chain Construction: The prompt, LLM, and output parser are combined into a chain using LangChain Expression Language (LCEL), creating a clear and executable workflow.
Personalized Patient Education:
The code simulates the process of retrieving a patient's details from the dimensional model.
The extracted information is then used to construct a personalized prompt for the LLM.
The LLM generates educational content tailored to the specific patient's needs, based on the provided context.
To run this code
Install Libraries:
bash
pip install pandas langchain langchain_openai
Use code with caution.

Set OpenAI API Key: Replace "YOUR_OPENAI_API_KEY" with your actual key or set it as an environment variable.
Execute the Script: Run the Python script.
This example highlights how a dimensional model can effectively organize healthcare data, enabling efficient extraction of features that are crucial for creating personalized and context-aware LLM solutions like the patient education system.
"""
