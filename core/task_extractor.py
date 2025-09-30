import json
from openai import OpenAI
from models.task import Task

class TaskExtractor:
    def __init__(self, client: OpenAI):
        self.client = client

    def extract(self, text: str):
        prompt = f"""
        You are a project assistant.
        From the following document text, extract tasks as a structured list.
        Return JSON with fields: id, title, description.
        
        Document:
        {text}
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            raw_tasks = json.loads(response.choices[0].message.content)
        except Exception:
            raw_tasks = []

        tasks = [Task(**task) for task in raw_tasks]
        return tasks
