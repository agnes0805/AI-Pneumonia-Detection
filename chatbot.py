import random


# -------------------------------
# Medical Knowledge Base
# -------------------------------

RESPONSES = {

    "hello": "Hello! 👋 I'm your AI Medical Assistant. Ask me anything about pneumonia or chest X-rays.",

    "hi": "Hi! 😊 How can I help you today?",

    "pneumonia":
        """Pneumonia is a lung infection that causes inflammation of the air sacs.
It may be caused by bacteria, viruses, or fungi. Common symptoms include cough,
fever, chest pain, and difficulty breathing.""",

    "symptoms":
        """Common symptoms of pneumonia include:
• Persistent cough
• Fever and chills
• Shortness of breath
• Chest pain while breathing
• Fatigue
• Rapid breathing""",

    "causes":
        """Pneumonia may be caused by:
• Bacteria
• Viruses
• Fungi
• Aspiration of food or liquids""",

    "treatment":
        """Treatment depends on the cause:
• Bacterial pneumonia → Antibiotics
• Viral pneumonia → Rest, fluids, antiviral medication if prescribed
• Severe cases may require hospitalization and oxygen therapy.""",

    "prevention":
        """Ways to reduce the risk of pneumonia:
• Get vaccinated
• Wash your hands frequently
• Avoid smoking
• Eat a balanced diet
• Exercise regularly
• Get enough sleep""",

    "covid":
        """COVID-19 can cause viral pneumonia in some patients.
Consult a healthcare professional if symptoms become severe.""",

    "xray":
        """Chest X-rays help doctors identify lung abnormalities such as
infection, fluid accumulation, or inflammation.""",

    "normal":
        """A normal chest X-ray generally shows clear lungs with no signs
of infection or fluid buildup.""",

    "medicine":
        """Only a qualified doctor should prescribe medicines.
Never self-medicate based solely on AI predictions.""",

    "doctor":
        """Seek medical attention immediately if you experience:
• Severe breathing difficulty
• High fever
• Persistent chest pain
• Bluish lips or face
• Confusion""",

    "thanks":
        "You're welcome! 😊 Stay healthy and take care.",

    "thank you":
        "Happy to help! Wishing you good health.",

    "bye":
        "Goodbye! 👋 Stay safe and consult a healthcare professional if needed."
}


DEFAULT_RESPONSES = [

    "I'm sorry, I don't have enough information about that. Please consult a healthcare professional.",

    "Could you please rephrase your question? I can answer questions about pneumonia, chest X-rays, symptoms, treatment, and prevention.",

    "I'm designed to answer pneumonia-related questions. Please ask about symptoms, causes, treatment, prevention, or chest X-rays."
]


def get_response(user_input):
    """
    Returns a chatbot response based on keyword matching.
    """

    if not user_input:
        return random.choice(DEFAULT_RESPONSES)

    question = user_input.lower()

    for keyword, answer in RESPONSES.items():

        if keyword in question:
            return answer

    return random.choice(DEFAULT_RESPONSES)