import json
import os
from pathlib import Path

import rich
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

env_path = Path(".env")
_ = load_dotenv(env_path)


class ContractSchema(BaseModel):
    category: str = Field(
        ..., description="Type of clause (e.g., Payment, Liability, Termination)"
    )
    risk_score: int = Field(..., description="Risk level from 1-10 (10 is high risk)")
    is_risky: bool = Field(..., description="True if risk_score > 5")
    reasoning: str = Field(..., description="Brief explanation of why this is risky")


client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ.get("GEMINI_API"),
)


base_system_prompt = """### INSTRUCTIONS:
Persona and capabilities: You are a legal advisor and an expert in law and contract understanding. Perform the following tasks:

1. Analyze the provided contract information.
2. Output a SINGLE JSON object that matches the schema below.
3. Do NOT output the schema definition itself.
4. Fill in the fields based on the contract information.

"""


def analyze_contract[T: BaseModel](
    llm_client: OpenAI, text: str, validation_schema: type[T]
) -> None | T:
    schema_definition: str = json.dumps(
        obj=validation_schema.model_json_schema(), indent=2
    )
    final_system_prompt = f"""
        {base_system_prompt}

        ### REQUIRED OUTPUT SCHEMA:
        {schema_definition}
        """
    print("Sending message to Gemini...")
    response = llm_client.chat.completions.create(
        model=os.environ.get("AI_MODEL"),
        messages=[
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": f"Analyze this clause: {text}"},
        ],
        response_format={"type": "json_object"},
    )

    try:
        raw_content = response.choices[0].message.content
        data: dict = dict()
        if raw_content is not None:
            data = json.loads(raw_content)

        if "properties" in data and "title" in data:
            print("Model failed: It just returned the schema definition.")
            return None

        validated_object = validation_schema(**data)
        return validated_object

    except Exception as e:
        print(f"Parsing Error: {e}")
        rich.print(response.choices[0].message.content)


if __name__ == "__main__":
    test_message = "The Tenant shall pay a penalty of 500% for any late payment."
    response = analyze_contract(client, test_message, ContractSchema)
    rich.print(response)
