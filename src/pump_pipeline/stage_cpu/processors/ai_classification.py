from __future__ import annotations
from typing import Any, Dict, Optional
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json

from src.core.processor import Processor
from src.pump_pipeline.pipeline_objects.data_object_pump import PumpPipelineDataObject


class AIRheemProcessor(Processor):
    """
    Collect artifacts into an AI-ready context and produce an AI-generated health artifact.

    Public behavior:
    - Reads existing artifacts from ``PumpPipelineDataObject`` (not normalized data)
    - Bundles them into a single ``ai_input_context`` artifact
    - Calls the OpenAI API
    """

    def validate_prerequisites(self, data_object: PumpPipelineDataObject) -> None:
        """Ensure we have a Rheem data object; artifacts are optional.

        Args:
            data_object (RheemPipelineDataObject): Pipeline data object.

        Raises:
            ValueError: If the provided data object is not Rheem-specific.
        """
        if not isinstance(data_object, PumpPipelineDataObject):
            raise ValueError(
                f"{self.name}: Expected PumpPipelineDataObject, got {type(data_object).__name__}"
            )

    def process(self, data_object: PumpPipelineDataObject) -> PumpPipelineDataObject:
        """Bundle existing artifacts for AI and set a placeholder output.

        Args:
            data_object (RheemPipelineDataObject): Current pipeline data object.

        Returns:
            RheemPipelineDataObject: Updated pipeline data object with AI artifacts.
        """
        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt(data_object)

        ai_response = self._call_ai_api(system_prompt, user_prompt)

        ai_device_health: Dict[str, Any] = {
            "signs_of_instability": ai_response["signs_of_instability"], 
            "description": ai_response["description"],  
            "current_state": ai_response["current_state"],  
        }
        data_object.set_artifact("ai_device_health", ai_device_health)

        return data_object



    def _system_prompt(self) -> str:
        """Return the system prompt for the AI pipeline."""
        return """
<task>
...
</task>

<use_case_context>
...
</use_case_context>
        """
    
    def _user_prompt(self, data_object: PumpPipelineDataObject) -> str:
        """Return the user prompt for the AI pipeline."""


        return f"""
...
        """
    
    def _call_ai_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call the AI API and return the parsed JSON response.

        This method loads environment variables (via ``dotenv``), constructs a
        wrapped OpenAI client using LangSmith's ``wrap_openai``, sends the
        provided system and user prompts, and requests a JSON-schema formatted
        response. The returned value is the parsed JSON object matching the
        schema described in the request.

        Returns:
            dict: Parsed JSON response from the AI model (keys: ``signs_of_instability``,
            ``description``, ``current_state``).

        Raises:
            RuntimeError: If the AI response cannot be parsed as JSON or does not
            contain the expected fields.
        """
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


        completion = client.responses.parse(
            model="gpt-5-mini", # "gpt-5-mini" "gpt-5-nano"
            reasoning={"effort": "low"},
            input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        )

        return completion.output_text
