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
    ai_model = os.environ["AI_MODEL"]
    schema_definition = json.dumps(validation_schema.model_json_schema(), indent=2)
    final_system_prompt = f"""
    ### INSTRUCTIONS
    {base_system_prompt}

    ### JSON OUTPUT STRUCTURE FORMAT
    {schema_definition}
    """
    print("Sending message to Gemini...")
    response = llm_client.chat.completions.parse(
        model=ai_model,
        messages=[
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": f"Analyze this clause: {text}"},
        ],
        response_format=validation_schema,
    )

    try:
        parsed_obj = response.choices[0].message.parsed
        
        if parsed_obj:
            return parsed_obj
        else:
            print("Model generated response with invalid structure.")
            return None

    except Exception as e:
        print(f"Agent Error: {e}")
        return None


if __name__ == "__main__":
    test_message = "The Tenant shall pay a penalty of 500% for any late payment."
    response = analyze_contract(client, test_message, ContractSchema)
    rich.print(response)
