# Claims Triage Model Service

Seeded synthetic claims, documented priority rules with label noise, stratified logistic-regression/random-forest comparison, a persisted model, and a Flask API returning probabilities and global feature importance. This is an educational ranking model.

## Run

Install requirements.txt, run `python model.py` to generate 20,000 claims, compare models and save the winner, then `flask --app app run`. `pytest` checks generation and input validation.
