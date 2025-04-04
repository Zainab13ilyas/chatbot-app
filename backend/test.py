import os
from openai import OpenAI

# Set your API key (ensure it's set as an environment variable)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# file=client.files.create(
#   file=open("dataset.jsonl", "rb"),
#   purpose="fine-tune"
# )
# print(file)

# tuned=client.fine_tuning.jobs.create(
#   training_file="file-5n2Q8Y9ZDcPTBDhEmHmDra",
#   model="gpt-3.5-turbo" #change to gpt-4-0613 if you have access
# )
# print(tuned)
#print(client.fine_tuning.jobs.list(limit=10))
job = client.fine_tuning.jobs.retrieve("ftjob-IQlBY5DYCc8PuSKvZnmdiZVW")
print(job.status)
