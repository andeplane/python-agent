from litellm import completion, ModelResponse, Choices
import litellm
from openai.types.chat import ChatCompletionMessageParam
from typing import Optional
import logging
import json

logger = logging.getLogger('llm')

# litellm.set_verbose=True
num_requests: int = 0

def chat_completion(
          message_history: list[ChatCompletionMessageParam], 
          prompt: str, 
          system_prompt: str | None = None,
          model: str = "gpt-4o-mini",
          ) -> Optional[str]:
        """
        Calls the OpenAI API with the given prompt and returns the response.

        Args:
            prompt (str): The prompt to send to the OpenAI API.

        Returns:
            Optional[str]: The API's response or None if failed.
        """
        global num_requests

        try:
            messages = [*message_history]
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = completion(model, messages=messages, temperature=0.7, max_tokens=1500, stream=False)
            assert(isinstance(response, ModelResponse))
            if response and response.choices:
                if len(response.choices) == 0:
                    raise ValueError("LLM response is empty.")
                
                if not isinstance(response.choices[0], Choices) or not response.choices[0].message or not response.choices[0].message.content:
                    raise ValueError("OpenAI API response is empty.")
                llm_response = response.choices[0].message.content.strip()
                num_requests += 1
                with open(f"requests/request_{num_requests}.txt", "w") as f:
                    f.write("Messages:\n")
                    json.dump(messages, f)
                    f.write("\n\nAnswer:\n")
                    json.dump(llm_response, f)
                # if self.debug:
                #     print(f"Prompt Sent:\n{prompt}\n")
                #     print(f"LLM Response:\n{llm_response}\n")
                return llm_response
        except Exception as e:
            print(f"OpenAI API call failed: {str(e)}")
        return None
    