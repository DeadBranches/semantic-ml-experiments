import vertexai
from vertexai.preview.language_models import TextGenerationModel


def predict_large_language_model_sample(
    project_id: str,
    model_name: str,
    temperature: float,
    max_decode_steps: int,
    top_p: float,
    top_k: int,
    content: str,
    location: str = "us-central1",
    tuned_model_name: str = "",
):
    """Predict using a Large Language Model."""
    vertexai.init(project=project_id, location=location)
    model = TextGenerationModel.from_pretrained(model_name)
    if tuned_model_name:
        model = model.get_tuned_model(tuned_model_name)
    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
        top_k=top_k,
        top_p=top_p,
    )
    print(f"Response from Model: {response.text}")


predict_large_language_model_sample(
    "ocr-project-382722",
    "text-bison@001",
    0.2,
    256,
    0.95,
    40,
    """Summarize the content of this email in a concise manner. The summary should be less than 2 short sentences.

Provide a laconic summary for the following conversation: PersonA: We need to take the accounts system offline to carry out the upgrade. But don\'t worry, it won\'t cause too much inconvenience. We\'re going to do it over the weekend.
PersonB: How long will the system be down for?
PersonA: We\'ll be taking everything offline in about two hours \' time. It\'ll be down for a minimum of twelve hours. If everything goes according to plan, it should be up again by 6 pm on Saturday.
PersonB: That\'s fine. We\'ve allowed forty-eight hours to be on the safe side.
Summary: The accounts system will be taken offline for a minimum of 12 hours starting in two hours to carry out an upgrade. It should be back up by 6 pm on Saturday.

Provide a laconic summary for the following conversation: Plato: Socrates, I have been thinking about what you said the other day about the importance of education.
Socrates: Yes, Plato, what about it?
Plato: Well, I was wondering if you could tell me more about how you think education can help people to live a good life.
Socrates: Of course. Education is the key to a good life. It is the process of learning how to think for yourself and how to make wise decisions. It is also the process of learning about the world around you and how to live in harmony with it.
Plato: I see. But how does one get an education?
Socrates: There are many ways to get an education. You can go to school, you can study on your own, or you can learn from the experiences of others. The important thing is to never stop learning.
Plato: I agree. But what do you think are the most important things to learn?
Socrates: The most important things to learn are how to think for yourself, how to make wise decisions, and how to live in harmony with the world around you.
Plato: I see. And how does one learn these things?
Socrates: You learn these things by questioning everything. You learn by asking questions about the world around you, about yourself, and about the meaning of life.
Plato: I see. And what do you think is the best way to question everything?
Socrates: The best way to question everything is to have conversations with people who have different opinions than you.
Plato: I see. And why is that?
Socrates: Because when you have conversations with people who have different opinions than you, you are forced to think about your own opinions. You are forced to defend your own beliefs. And in doing so, you learn more about yourself and about the world around you.
Summary: Socrates believes that education is the key to a good life. It is the process of learning how to think for yourself, how to make wise decisions, and how to live in harmony with the world around you. The most important things to learn are how to think for yourself, how to make wise decisions, and how to live in harmony with the world around you. You learn these things by questioning everything, and the best way to question everything is to have conversations with people who have different opinions than you.

Provide a laconic summary for the following conversation: Alice: Hey, Bob, what are some of your ideas for the team morale event?
Bob: Everyone seemed to enjoy the potluck and board game that we did last time.
Alice: I think so too.
Bob: Maybe we can do something similar but at a different location?
Alice: That sounds good. Where did you have in mind?
Bob: I was thinking we could reserve the picnic area at Sunset Beach Park.
Alice: Good idea. In addition to board games, we could also bring a frisbee and volleyball for the beach.
Bob: Perfect! Let me make the reservation now.
Summary: Alice and Bob are planning a team morale event. They are considering having a potluck and board games at Sunset Beach Park.

Provide a laconic summary for the following conversation: Accommodation Plan Request
 
Kat Green <kat@katgreen.ca> Mon, Jan 31, 2022 at 3:35 PM To: Paige Halam-Andres <phalamandres@gmail.com>, Nicholas Halam-Andres <nicholas.halaman@gmail.com> 
Hi Nick and Paige, 
First, thanks for respecting the pause in communication I requested. It’s much appreciated. 
I’d like to address the main substantive issue and source of conflict between us. You’ve both expressed that you want the exhaust fan that is currently 
mounted in the bathroom window (using dryer ducting, cardboard, and tape) removed. Let’s figure out how to accomplish this. 
Please provide an accommodation plan that introduces measures to reduce the risk of airborne virus transmission that can occur in an apartment building. The measures I’m asking you to propose are a replacement for the existing exhaust fan. I’m requesting this accommodation plan because, as you know, I live with a disability, am immunocompromised as a result of treatment, and as such I require ventilation in my unit to reduce the risk of serious illness from covid-19. 
I’ve been advised to specify a timeline for your response and also for the requested accommodation plan. Please respond to me using email by February 1st, at 3:30pm, which is 24 hours. Regarding a timeline for the accommodation plan, I don’t know what a reasonable amount of time is. That being said, given that accommodation plans are intended to be implemented as quickly as possible I’m requesting that you please provide a plan by February 7, 2020 at 3:30pm (7 days). It’s acceptable to me if you share the plan with me by email or as a hard copy at my back door (where I receive mail). 
Best, Kat Green Urban Planner 
w: katgreen.ca 
4/5/2022, 2:39 PM 
1 of 1 
Summary:
""",
    "us-central1",
)
