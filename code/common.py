"""
Common data structures and base classes for CG optimization agents
"""

import json
import json_repair
import requests
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class AgentRole(Enum):
    MAPPING = "mapping"
    TOPOLOGY = "topology"
    BOUNDARY = "boundary"
    DIAGNOSTIC = "diagnostic"
    HYPOTHESIS = "hypothesis"
    OPTIMIZATION = "optimization"


@dataclass
class MappingScheme:
    """Defines the CG mapping scheme"""
    bead_types: List[str]
    bead_descriptions: Dict[str, str]
    connectivity: List[tuple]
    dummy_beads: List[str]
    interaction_matrix: Dict[str, List[str]]


@dataclass
class ParameterBoundary:
    """Parameter ranges and physical constraints"""
    var_names: List[str]
    min_var: List[float]
    max_var: List[float]
    recommended_start: List[float]
    physical_constraints: Dict[str, str]


@dataclass
class DiagnosticReport:
    """System diagnostic results"""
    iteration: int
    phase_state: str  # "liquid", "gas", "solid", "unstable"
    density_assessment: str
    hvap_assessment: str
    surface_tension_assessment: str
    warnings: List[str]
    recommendations: List[str]
    confidence_score: float
    boundary_adjustment: Optional[ParameterBoundary] = None


@dataclass
class OptimizationState:
    """Current optimization state"""
    iteration: int
    phase: str
    best_score: float
    best_params: Dict[str, float]
    recent_scores: List[float]
    stuck_counter: int
    crash_regions: List[Dict[str, float]]


class LLMAgent:
    """Base class for LLM agents"""

    def __init__(self, role: AgentRole, api_key: str, url: str):
        self.role = role
        self.api_key = api_key
        self.url = url
        self.history = []

    def call(self, prompt: str, system_prompt: Optional[str] = None,
              max_retries: int = 3, temperature: float = 0.7) -> Optional[Dict]:
        """Make API call to LLM"""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        if "completions" in self.url and "chat" not in self.url:
            # Use completions API
            # Build prompt from messages
            prompt_str = ""
            if system_prompt:
                prompt_str += f"{system_prompt}\n\n"
            # Add conversation history
            for msg in self.history:
                role = msg['role']
                content = msg['content']
                if role == "system":
                    prompt_str += f"{content}\n\n"
                elif role == "user":
                    prompt_str += f"{content}\n\n"
                elif role == "assistant":
                    prompt_str += f"{content}\n\n"
            prompt_str += f"{prompt}"

            for attempt in range(max_retries):
                try:
                    payload = {
                        "prompt": prompt_str,
                        "max_tokens": 512,  # Reduce max tokens
                        "temperature": temperature,
                        "top_p": 0.9,
                        "seed": random.randint(0, 2**32 - 1)
                    }

                    response = requests.post(self.url, headers=headers, json=payload, timeout=300)
                    response.raise_for_status()
                    response_data = response.json()

                    if not response_data or "choices" not in response_data:
                        print(f"[{self.role.value}] Invalid response structure")
                        continue

                    response_text = response_data["choices"][0].get("text", "")
                    if not response_text:
                        print(f"[{self.role.value}] Empty response")
                        continue

                    # Clean up response
                    if response_text.strip().startswith("```json"):
                        response_text = response_text.strip()[7:]
                        if response_text.endswith("```"):
                            response_text = response_text[:-3]

                    # Parse JSON
                    try:
                        result = json.loads(response_text.strip())
                        # Update history
                        self.history.append({"role": "user", "content": prompt})
                        self.history.append({"role": "assistant", "content": response_text})
                        return result
                    except json.JSONDecodeError as e:
                        print(f"[{self.role.value}] JSON parse error: {e}")
                        print(f"[{self.role.value}] Response text: {response_text[:500]}")
                        continue

                except Exception as e:
                    print(f"[{self.role.value}] Error on attempt {attempt+1}: {e}")
                    continue

        else:
            # Use chat API
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add conversation history
            for msg in self.history:
                messages.append({"role": msg['role'], "content": msg['content']})

            messages.append({"role": "user", "content": prompt})

            for attempt in range(max_retries):
                try:
                    payload = {
                        #"model": "gpt-oss-120b",
                        "model": "Kimi-K2.5",
                        "messages": messages,
                        "max_tokens": 16384,  
                        "temperature": temperature,
                        "top_p": 0.9,
                        "seed": random.randint(0, 2**32 - 1),
                        "presence_penalty": 0.1,   # Encourage topic diversity
                        "num_ctx": 131072,         # Explicitly set context window (if supported) 128K
                    }

                    response = requests.post(self.url, headers=headers, json=payload, timeout=300)
                    response.raise_for_status()
                    response_data = response.json()

                    if not response_data or "choices" not in response_data:
                        print(f"[{self.role.value}] Invalid response structure")
                        continue

                    response_text = response_data["choices"][0].get("message", {}).get("content", "")
                    if not response_text:
                        print(f"[{self.role.value}] Empty response")
                        continue

                    # Clean up response
                    if response_text.strip().startswith("```json"):
                        response_text = response_text.strip()[7:]
                        if response_text.endswith("```"):
                            response_text = response_text[:-3]

                    # print(f"[{self.role.value}] Raw response:")
                    # print(response_text.strip())

                    # Parse JSON
                    try:
                        result = json.loads(response_text.strip())
                        # Update history
                        #self.history.append({"role": "user", "content": prompt})
                        #self.history.append({"role": "assistant", "content": response_text})
                        return result
                    except json.JSONDecodeError as e:
                        print(f"[{self.role.value}] JSON parse error: {e}")
                        print(f"[{self.role.value}] Response text: {response_text[:500]}")
                        continue

                except Exception as e:
                    print(f"[{self.role.value}] Error on attempt {attempt+1}: {e}")
                    continue

        print(f"[{self.role.value}] Failed after {max_retries} attempts")
        return None

    def reset_history(self):
        """Clear conversation history"""
        self.history = []